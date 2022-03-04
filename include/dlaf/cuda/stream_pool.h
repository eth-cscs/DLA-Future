//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#ifdef DLAF_WITH_CUDA

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <pika/modules/concurrency.hpp>
#include <pika/runtime.hpp>
#include <pika/thread.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/cuda/error.h"

namespace dlaf {
namespace cuda {
namespace internal {

struct StreamPoolImpl {
  int device_;
  std::size_t num_worker_threads_ = pika::get_num_worker_threads();
  std::size_t num_streams_per_worker_thread_;
  std::vector<cudaStream_t> streams_;
  std::vector<pika::util::cache_aligned_data<std::size_t>> current_stream_idxs_;

  StreamPoolImpl(int device, std::size_t num_streams_per_worker_thread,
                 pika::threads::thread_priority pika_thread_priority)
      : device_(device), num_streams_per_worker_thread_(num_streams_per_worker_thread),
        streams_(num_worker_threads_ * num_streams_per_worker_thread),
        current_stream_idxs_(num_worker_threads_, {std::size_t(0)}) {
    DLAF_CUDA_CALL(cudaSetDevice(device));

    // We map pika::threads::thread_priority::high to the highest CUDA stream
    // priority, and the rest to the lowest. Typically CUDA streams will only
    // have two priorities.
    int least_priority, greatest_priority;
    DLAF_CUDA_CALL(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    int stream_priority = least_priority;
    if (pika_thread_priority == pika::threads::thread_priority::high) {
      stream_priority = greatest_priority;
    }

    for (auto& s : streams_) {
      DLAF_CUDA_CALL(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, stream_priority));
    }
  }

  StreamPoolImpl& operator=(StreamPoolImpl&&) = default;
  StreamPoolImpl(StreamPoolImpl&&) = default;
  StreamPoolImpl(const StreamPoolImpl&) = delete;
  StreamPoolImpl& operator=(const StreamPoolImpl&) = delete;

  ~StreamPoolImpl() {
    for (auto& s : streams_) {
      DLAF_CUDA_CALL(cudaStreamDestroy(s));
    }
  }

  cudaStream_t getNextStream() {
    // Set the device corresponding to the CUBLAS handle.
    //
    // The CUBLAS library context is tied to the current CUDA device [1]. A previous task scheduled on
    // the same thread may have set a different device, this makes sure the correct device is used. The
    // function is considered very low overhead call [2].
    //
    // [1]: https://docs.nvidia.com/cuda/cublas/index.html#cublascreate
    // [2]: CUDA Runtime API, section 5.1 Device Management
    DLAF_CUDA_CALL(cudaSetDevice(device_));
    const std::size_t worker_thread_num = pika::get_worker_thread_num();
    DLAF_ASSERT(worker_thread_num != std::size_t(-1), worker_thread_num);
    std::size_t stream_idx =
        worker_thread_num * num_streams_per_worker_thread_ +
        (++current_stream_idxs_[worker_thread_num].data_ % num_streams_per_worker_thread_);

    return streams_[stream_idx];
  }

  int getDevice() {
    return device_;
  }
};
}

/// A pool of CUDA streams with reference semantics (copying points to the same
/// underlying CUDA streams, last reference destroys the references).  Allows
/// access to CUDA streams in a round-robin fashion.  Each pika worker thread is
/// assigned a set of thread local CUDA streams.
class StreamPool {
  std::shared_ptr<internal::StreamPoolImpl> streams_ptr_;

public:
  StreamPool(
      int device = 0, std::size_t num_streams_per_worker_thread = 3,
      pika::threads::thread_priority pika_thread_priority = pika::threads::thread_priority::default_)
      : streams_ptr_(std::make_shared<internal::StreamPoolImpl>(device, num_streams_per_worker_thread,
                                                                pika_thread_priority)) {}

  cudaStream_t getNextStream() {
    DLAF_ASSERT(bool(streams_ptr_), "");
    return streams_ptr_->getNextStream();
  }

  int getDevice() {
    DLAF_ASSERT(bool(streams_ptr_), "");
    return streams_ptr_->getDevice();
  }

  bool operator==(StreamPool const& rhs) const noexcept {
    return streams_ptr_ == rhs.streams_ptr_;
  }

  bool operator!=(StreamPool const& rhs) const noexcept {
    return !(*this == rhs);
  }
};
}
}

#endif
