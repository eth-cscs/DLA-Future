//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <hpx/future.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/thread_executors.hpp>
#include <hpx/modules/execution_base.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/cuda/error.h"
#include "dlaf/cuda/event.h"
#include "dlaf/cuda/mutex.h"

namespace dlaf {
namespace cuda {

// Print information about the devices.
inline void print_device_info(int device) noexcept {
  cudaDeviceProp device_prop;
  DLAF_CUDA_CALL(cudaGetDeviceProperties(&device_prop, device));
  printf("Device %d has compute capability %d.%d.\n", device, device_prop.major, device_prop.minor);
}

// Return the number of devices.
inline int num_devices() noexcept {
  int ndevices;
  cudaGetDeviceCount(&ndevices);
  return ndevices;
}

namespace internal {

// This class works around the fact std::shared_ptr doesn't support raw arrays in C++14.
//
class streams_array {
  cudaStream_t* arr_;
  int num_streams_;

public:
  streams_array(int device, int num_streams) noexcept
      : arr_(new cudaStream_t[num_streams]), num_streams_(num_streams) {
    DLAF_CUDA_CALL(cudaSetDevice(device));
    for (int i = 0; i < num_streams; ++i) {
      DLAF_CUDA_CALL(cudaStreamCreateWithFlags(&(arr_[i]), cudaStreamNonBlocking));
    }
  }

  streams_array& operator=(streams_array&& o) noexcept {
    arr_ = o.arr_;
    num_streams_ = o.num_streams_;
    o.arr_ = nullptr;
    return *this;
  }

  streams_array(streams_array&& o) noexcept {
    *this = std::move(o);
  }

  streams_array(const streams_array&) = delete;

  streams_array& operator=(const streams_array&) = delete;

  ~streams_array() {
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

}

/// An executor for CUDA calls.
///
/// Note: The streams are rotated in Round-robin.
class executor {
protected:
  int device_;
  std::shared_ptr<internal::streams_array> streams_ptr_;
  hpx::threads::executors::pool_executor threads_executor_;
  int curr_stream_idx_;

public:
  // Associate the parallel_execution_tag executor tag type as a default with this executor.
  using execution_category = hpx::parallel::execution::parallel_execution_tag;

  executor(int device, int num_streams)
      : device_(device), streams_ptr_(std::make_shared<internal::streams_array>(device, num_streams)),
        threads_executor_("default", hpx::threads::thread_priority_high), curr_stream_idx_(0) {}

  bool operator==(executor const& rhs) const noexcept {
    return streams_ptr_ == rhs.streams_ptr_ && threads_executor_ == rhs.threads_executor_;
  }

  bool operator!=(executor const& rhs) const noexcept {
    return !(*this == rhs);
  }

  executor const& context() const noexcept {
    return *this;
  }

  executor& set_curr_stream_idx(int curr_stream_idx) noexcept {
    DLAF_ASSERT(curr_stream_idx >= 0, curr_stream_idx);
    DLAF_ASSERT(curr_stream_idx < streams_ptr_->size(), curr_stream_idx);
    curr_stream_idx_ = curr_stream_idx;
    return *this;
  }

  // Implement the TwoWayExecutor interface.
  //
  // Note: the member can't be `const` because of `threads_executor_`.
  // Note: Parameters are passed by value as they are small types: pointers, integers or scalars.
  template <typename F, typename... Ts>
  hpx::future<void> async_execute(F f, Ts... ts) {
    // Set the device to associate the following event (`ev`) with it. [1]
    //
    // A previous task scheduled on the same thread may have set a different device, this makes sure the
    // correct device is used. The function is considered very low overhead [2]. Any previous assignment
    // of CUDA devices to threads is not preserved.
    //
    // [1]: CUDA Programming Guide, section 3.2.7.2 Device selection
    // [2]: CUDA Runtime API, section 5.1 Device Management
    DLAF_CUDA_CALL(cudaSetDevice(device_));

    // Use an event to query the CUDA kernel for completion. Timing is disabled for performance. [1]
    //
    // [1]: CUDA Runtime API, section 5.5 Event Management
    cuda::Event ev{};

    // Call the CUDA function `f` and schedule an event after it.
    //
    // The event indicates the the function `f` has completed. The stream may be shared by mutliple
    // host threads, the mutex is here to make sure no other CUDA calls or events are scheduled
    // between the call to `f` and it's corresponding event.
    {
      std::lock_guard<hpx::lcos::local::mutex> lk(internal::get_cuda_mtx());
      cudaStream_t stream = (*streams_ptr_)[curr_stream_idx_];
      DLAF_CUDA_CALL(f(ts..., stream));
      ev.record(stream);
      curr_stream_idx_ = (curr_stream_idx_ + 1) % streams_ptr_->size();
    }
    return hpx::async(threads_executor_, [e = std::move(ev)] { e.query(); });
  }
};

}
}

namespace hpx {
namespace parallel {
namespace execution {

template <>
struct is_two_way_executor<dlaf::cuda::executor> : std::true_type {};

}
}
}
