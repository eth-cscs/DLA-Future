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

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <hpx/execution.hpp>
#include <hpx/functional.hpp>
#include <hpx/future.hpp>
#include <hpx/modules/async_cuda.hpp>
#include <hpx/mutex.hpp>
#include <hpx/tuple.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/cublas/error.h"
#include "dlaf/cuda/error.h"
#include "dlaf/cuda/executor.h"

namespace dlaf {
namespace cublas {
namespace internal {
class HandlePoolImpl {
  int device_;
  std::size_t num_worker_threads_ = hpx::get_num_worker_threads();
  std::vector<cublasHandle_t> handles_;
  cublasPointerMode_t ptr_mode_;

public:
  HandlePoolImpl(int device, cublasPointerMode_t ptr_mode)
      : device_(device), handles_(num_worker_threads_), ptr_mode_(ptr_mode) {
    DLAF_CUDA_CALL(cudaSetDevice(device_));

    for (auto& h : handles_) {
      DLAF_CUBLAS_CALL(cublasCreate(&h));
    }
  }

  HandlePoolImpl& operator=(HandlePoolImpl&&) = default;
  HandlePoolImpl(HandlePoolImpl&&) = default;
  HandlePoolImpl(const HandlePoolImpl&) = delete;
  HandlePoolImpl& operator=(const HandlePoolImpl&) = delete;

  ~HandlePoolImpl() {
    for (auto& h : handles_) {
      DLAF_CUBLAS_CALL(cublasDestroy(h));
    }
  }

  cublasHandle_t getNextHandle(cudaStream_t stream) {
    cublasHandle_t handle = handles_[hpx::get_worker_thread_num()];
    DLAF_CUDA_CALL(cudaSetDevice(device_));
    DLAF_CUBLAS_CALL(cublasSetStream(handle, stream));
    DLAF_CUBLAS_CALL(cublasSetPointerMode(handle, ptr_mode_));
    return handle;
  }

  int getDevice() {
    return device_;
  }
};
}

/// A pool of cuBLAS handles with reference semantics (copying points to the
/// same underlying cuBLAS handles, last reference destroys the references).
/// Allows access to cuBLAS handles associated with a particular stream. The
/// user must ensure that the handle pool and the stream use the same device.
/// Each HPX worker thread is assigned thread local cuBLAS handle.
class HandlePool {
  std::shared_ptr<internal::HandlePoolImpl> handles_ptr_;

public:
  HandlePool(int device = 0, cublasPointerMode_t ptr_mode = CUBLAS_POINTER_MODE_HOST)
      : handles_ptr_(std::make_shared<internal::HandlePoolImpl>(device, ptr_mode)) {}

  cublasHandle_t getNextHandle(cudaStream_t stream) {
    DLAF_ASSERT(bool(handles_ptr_), "");
    return handles_ptr_->getNextHandle(stream);
  }

  int getDevice() {
    DLAF_ASSERT(bool(handles_ptr_), "");
    return handles_ptr_->getDevice();
  }

  bool operator==(HandlePool const& rhs) const noexcept {
    return handles_ptr_ == rhs.handles_ptr_;
  }

  bool operator!=(HandlePool const& rhs) const noexcept {
    return !(*this == rhs);
  }
};

/// An executor for cuBLAS functions. Uses handles and streams from the given
/// HandlePool and StreamPool. A cuBLAS function is defined as any function that
/// takes a cuBLAS handle as the first argument. The executor inserts a cuBLAS
/// handle into the argument list, i.e. a handle should not be provided at the
/// apply/async/dataflow invocation site.
class Executor : public cuda::Executor {
  using base = cuda::Executor;
  HandlePool handle_pool_;

public:
  Executor(cuda::StreamPool stream_pool, HandlePool handle_pool)
      : base(stream_pool), handle_pool_(handle_pool) {
    DLAF_ASSERT(stream_pool_.getDevice() == handle_pool_.getDevice(), "", stream_pool_.getDevice(),
                handle_pool_.getDevice());
  }

  bool operator==(Executor const& rhs) const noexcept {
    return base::operator==(rhs) && handle_pool_ == rhs.handle_pool_;
  }

  bool operator!=(Executor const& rhs) const noexcept {
    return !(*this == rhs);
  }

  Executor const& context() const noexcept {
    return *this;
  }

  template <typename F, typename... Ts>
  auto async_execute(F&& f, Ts&&... ts) {
    cudaStream_t stream = stream_pool_.getNextStream();
    cublasHandle_t handle = handle_pool_.getNextHandle(stream);
    auto r = hpx::invoke(std::forward<F>(f), handle, std::forward<Ts>(ts)...);
    hpx::future<void> fut = hpx::cuda::experimental::detail::get_future_with_event(stream);

    return fut.then(hpx::launch::sync,
                    [r = std::move(r)](hpx::future<void>&&) mutable { return std::move(r); });
  }

  template <class Frame, class F, class Futures>
  void dataflow_finalize(Frame&& frame, F&& f, Futures&& futures) {
    // Ensure the dataflow frame stays alive long enough.
    hpx::intrusive_ptr<typename std::remove_pointer<typename std::decay<Frame>::type>::type> frame_p(
        frame);

    cudaStream_t stream = stream_pool_.getNextStream();
    cublasHandle_t handle = handle_pool_.getNextHandle(stream);
    auto r = hpx::invoke_fused(std::forward<F>(f),
                               hpx::tuple_cat(hpx::tie(handle), std::forward<Futures>(futures)));
    hpx::future<void> fut = hpx::cuda::experimental::detail::get_future_with_event(stream);

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
struct is_two_way_executor<dlaf::cublas::Executor> : std::true_type {};
}
}
}

#endif
