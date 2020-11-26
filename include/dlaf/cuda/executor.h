//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2020, ETH Zurich
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

#include <hpx/execution.hpp>
#include <hpx/functional.hpp>
#include <hpx/future.hpp>
#include <hpx/mutex.hpp>
#include <hpx/tuple.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/cuda/error.h"

namespace dlaf {
namespace cuda {
namespace internal {

// This class works around the fact std::shared_ptr doesn't support raw arrays
// in C++14.
class StreamArray {
  cudaStream_t* arr_;
  int num_streams_;

public:
  StreamArray(int device, int num_streams, hpx::threads::thread_priority hpx_thread_priority) noexcept
      : arr_(new cudaStream_t[num_streams]), num_streams_(num_streams) {
    DLAF_CUDA_CALL(cudaSetDevice(device));

    // We map hpx::threads::thread_priority_high to the highest CUDA stream
    // priority, and the rest to the lowest. Typically CUDA streams will only
    // have two priorities.
    int least_priority, greatest_priority;
    DLAF_CUDA_CALL(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    int stream_priority = least_priority;
    if (hpx_thread_priority == hpx::threads::thread_priority_high) {
      stream_priority = greatest_priority;
    }

    for (int i = 0; i < num_streams; ++i) {
      DLAF_CUDA_CALL(cudaStreamCreateWithPriority(&(arr_[i]), cudaStreamNonBlocking, stream_priority));
    }
  }

  StreamArray& operator=(StreamArray&& o) noexcept {
    arr_ = o.arr_;
    num_streams_ = o.num_streams_;
    o.arr_ = nullptr;
    return *this;
  }

  StreamArray(StreamArray&& o) noexcept {
    *this = std::move(o);
  }

  StreamArray(const StreamArray&) = delete;

  StreamArray& operator=(const StreamArray&) = delete;

  ~StreamArray() {
    if (arr_ == nullptr)
      return;

    for (int i = 0; i < num_streams_; ++i) {
      DLAF_CUDA_CALL(cudaStreamDestroy(arr_[i]));
    }
    delete[] arr_;
  }

  cudaStream_t operator[](int i) const noexcept {
    return arr_[i];
  }

  int size() const noexcept {
    return num_streams_;
  }
};

// Helper class which holds a CUDA stream, protected by a lock. This class is
// intended to be used as an RAII guard. The usage of the stream is protected
// as long as the instance is alive. It also provides a means to get an event
// corresponding to the stream. Instances of this object can not be shared
// among threads. The lifetime of an instance must be no longer than the
// StreamPool from which an instance was taken.
class LockedStream {
  cudaStream_t stream_;
  std::unique_lock<hpx::mutex> lk_;

public:
  LockedStream(cudaStream_t stream, std::unique_lock<hpx::mutex> lk)
      : stream_(stream), lk_(std::move(lk)) {}

  void unlock() {
    lk_.unlock();
  };

  cudaStream_t& getStream() {
    return stream_;
  }
};

// Helper class with a reference counted array of CUDA streams. Allows access
// to RAII locked streams (LockedStream). Ensures that the correct device is
// set.
class StreamPool {
  int device_;
  std::size_t curr_stream_idx_ = 0;
  std::shared_ptr<StreamArray> streams_ptr_;
  std::shared_ptr<hpx::mutex> mtx_;

public:
  StreamPool(int device, int num_streams, hpx::threads::thread_priority priority)
      : device_(device), streams_ptr_(std::make_shared<StreamArray>(device, num_streams, priority)),
        mtx_(std::make_shared<hpx::mutex>()) {}

  LockedStream getNextStream() {
    // Set the device corresponding to the CUBLAS handle.
    //
    // The CUBLAS library context is tied to the current CUDA device [1]. A previous task scheduled on
    // the same thread may have set a different device, this makes sure the correct device is used. The
    // function is considered very low overhead call [2].
    //
    // [1]: https://docs.nvidia.com/cuda/cublas/index.html#cublascreate
    // [2]: CUDA Runtime API, section 5.1 Device Management
    DLAF_CUDA_CALL(cudaSetDevice(device_));

    curr_stream_idx_ = (curr_stream_idx_ + 1) % streams_ptr_->size();
    return LockedStream((*streams_ptr_)[curr_stream_idx_], std::unique_lock<hpx::mutex>(*mtx_.get()));
  }

  bool operator==(StreamPool const& rhs) const noexcept {
    return streams_ptr_ == rhs.streams_ptr_;
  }

  bool operator!=(StreamPool const& rhs) const noexcept {
    return !(*this == rhs);
  }
};
}

/// An executor for CUDA calls.
///
/// Note: The streams are rotated in Round-robin.
class Executor {
protected:
  internal::StreamPool stream_pool_;
  hpx::threads::thread_priority priority_ = hpx::threads::thread_priority_normal;

public:
  Executor(int device, int num_streams,
           hpx::threads::thread_priority priority = hpx::threads::thread_priority_normal)
      : stream_pool_(device, num_streams, priority), priority_(priority) {}

  bool operator==(Executor const& rhs) const noexcept {
    return stream_pool_ == rhs.stream_pool_;
  }

  bool operator!=(Executor const& rhs) const noexcept {
    return !(*this == rhs);
  }

  Executor const& context() const noexcept {
    return *this;
  }

  template <typename F, typename... Ts>
  auto async_execute(F&& f, Ts&&... ts) {
    internal::LockedStream locked_stream = stream_pool_.getNextStream();
    auto r = hpx::invoke(std::forward<F>(f), std::forward<Ts>(ts)..., locked_stream.getStream());
    hpx::future<void> fut =
        hpx::cuda::experimental::detail::get_future_with_event(locked_stream.getStream());
    locked_stream.unlock();

    return fut.then(hpx::launch::sync,
                    [r = std::move(r)](hpx::future<void>&&) mutable { return std::move(r); });
  }

  template <class Frame, class F, class Futures>
  void dataflow_finalize(Frame&& frame, F&& f, Futures&& futures) {
    // Ensure the dataflow frame stays alive long enough.
    hpx::intrusive_ptr<typename std::remove_pointer<typename std::decay<Frame>::type>::type> frame_p(
        frame);

    internal::LockedStream locked_stream = stream_pool_.getNextStream();
    auto r = hpx::invoke_fused(std::forward<F>(f), hpx::tuple_cat(std::forward<Futures>(futures),
                                                                  hpx::tie(locked_stream.getStream())));
    hpx::future<void> fut =
        hpx::cuda::experimental::detail::get_future_with_event(locked_stream.getStream());
    locked_stream.unlock();

    fut.then(hpx::launch::sync, [r = std::move(r), frame_p = std::move(frame_p)](
                                    hpx::future<void>&&) mutable { frame_p->set_data(std::move(r)); });
  }
};
}
}

namespace hpx {
namespace parallel {
namespace execution {

template <>
struct is_two_way_executor<dlaf::cuda::Executor> : std::true_type {};
}
}
}

#endif
