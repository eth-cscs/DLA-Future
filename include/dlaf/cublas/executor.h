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

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <hpx/future.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/thread_executors.hpp>
#include <hpx/modules/execution_base.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/cublas/error.h"
#include "dlaf/cuda/error.h"
#include "dlaf/cuda/event.h"
#include "dlaf/cuda/executor.h"
#include "dlaf/cuda/mutex.h"

namespace dlaf {
namespace cublas {

namespace internal {

struct cublas_handle_wrapper {
  cublasHandle_t handle;
  cublas_handle_wrapper(int device) noexcept {
    DLAF_CUDA_CALL(cudaSetDevice(device));
    DLAF_CUBLAS_CALL(cublasCreate(&handle));
  }
  ~cublas_handle_wrapper() {
    // This implicitly calls `cublasDeviceSynchronize()` [1].
    //
    // [1]: cuBLAS, section 2.4 cuBLAS Helper Function Reference
    DLAF_CUBLAS_CALL(cublasDestroy(handle));
  }
};
}

/// An executor for a CUBLAS functions. Each device has a single CUBLAS handle associated to it.
class executor : public cuda::executor {
  using base = cuda::executor;

  std::shared_ptr<internal::cublas_handle_wrapper> handler_ptr_;
  cublasPointerMode_t cublas_ptr_mode_;

public:
  // Associate the parallel_execution_tag executor tag type as a default with this executor.
  using execution_category = hpx::parallel::execution::parallel_execution_tag;

  executor(int device, int num_streams, cublasPointerMode_t ptr_mode)
      : base(device, num_streams),
        handler_ptr_(std::make_shared<internal::cublas_handle_wrapper>(device)),
        cublas_ptr_mode_(ptr_mode) {}

  executor(cuda::executor ex, cublasPointerMode_t ptr_mode)
      : base(std::move(ex)),
        handler_ptr_(std::make_shared<internal::cublas_handle_wrapper>(base::device_)),
        cublas_ptr_mode_(ptr_mode) {}

  bool operator==(executor const& rhs) const noexcept {
    return base::operator==(rhs) && handler_ptr_ == rhs.handler_ptr_ &&
           cublas_ptr_mode_ == rhs.cublas_ptr_mode_;
  }

  bool operator!=(executor const& rhs) const noexcept {
    return !(*this == rhs);
  }

  executor const& context() const noexcept {
    return *this;
  }

  executor& reset_ptr_mode(cublasPointerMode_t ptr_mode) noexcept {
    cublas_ptr_mode_ = ptr_mode;
    return *this;
  }

  // Implement the TwoWayExecutor interface.
  //
  // Note: the member can't be marked `const` because of `threads_executor_`.
  // Note: Parameters are passed by value as they are small types: pointers, integers or scalars.
  template <class Return, class... Params, class... Ts>
  hpx::future<typename std::enable_if<std::is_same<cublasStatus_t, Return>::value, void>::type>
  async_execute(Return (*cublas_function)(Params...), Ts... ts) {
    // Set the device corresponding to the CUBLAS handle.
    //
    // The CUBLAS library context is tied to the current CUDA device [1]. A previous task scheduled on
    // the same thread may have set a different device, this makes sure the correct device is used. The
    // function is considered very low overhead call [2].
    //
    // [1]: https://docs.nvidia.com/cuda/cublas/index.html#cublascreate
    // [2]: CUDA Runtime API, section 5.1 Device Management
    DLAF_CUDA_CALL(cudaSetDevice(base::device_));

    // Use an event to query the CUBLAS kernel for completion. Timing is disabled for performance. [1]
    //
    // [1]: CUDA Runtime API, section 5.5 Event Management
    cuda::Event ev{};

    // Call the CUBLAS function `f` and schedule an event after it.
    //
    // The event indicates the the function `f` has completed. The handle may be shared by mutliple
    // host threads, the mutex is here to make sure no other CUBLAS calls or events are scheduled
    // between the call to `f` and it's corresponding event.
    {
      std::lock_guard<hpx::lcos::local::mutex> lk(cuda::internal::get_cuda_mtx());
      cudaStream_t stream = (*base::streams_ptr_)[base::curr_stream_idx_];
      cublasHandle_t handle = handler_ptr_->handle;
      DLAF_CUBLAS_CALL(cublasSetStream(handle, stream));
      DLAF_CUBLAS_CALL(cublasSetPointerMode(handle, cublas_ptr_mode_));
      DLAF_CUBLAS_CALL(cublas_function(handle, ts...));
      ev.record(stream);
      base::curr_stream_idx_ = (base::curr_stream_idx_ + 1) % base::streams_ptr_->size();
    }
    return hpx::async(base::threads_executor_, [e = std::move(ev)] { e.query(); });
  }

  template <class Return, class... Params, class... Args>
  hpx::future<typename std::enable_if<std::is_same<cudaError_t, Return>::value, void>::type> async_execute(
      Return (*cuda_function)(Params...), Args... args) {
    return base::async_execute(cuda_function, args...);
  }
};
}
}

namespace hpx {
namespace parallel {
namespace execution {

template <>
struct is_two_way_executor<dlaf::cublas::executor> : std::true_type {};

}
}
}
