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

#include "dlaf/cublas/error.h"
#include "dlaf/cuda/error.h"
#include "dlaf/cuda/executor.h"

namespace dlaf {
namespace cublas {
// Helper class for a cuBLAS handle associated with a CUDA stream.
class StreamHandle {
  cudaStream_t stream_;
  cublasHandle_t handle_;

public:
  StreamHandle(cudaStream_t stream, cublasHandle_t handle) : stream_(stream), handle_(handle) {}

  cublasHandle_t& getHandle() {
    return handle_;
  }

  cudaStream_t& getStream() {
    return stream_;
  }
};

namespace internal {
class HandlePoolImpl {
  std::size_t num_worker_threads_ = hpx::get_num_worker_threads();
  cuda::StreamPool stream_pool_;
  std::vector<cublasHandle_t> handles_;
  cublasPointerMode_t ptr_mode_;

public:
  HandlePoolImpl(cuda::StreamPool stream_pool, cublasPointerMode_t ptr_mode)
      : stream_pool_(stream_pool), handles_(num_worker_threads_), ptr_mode_(ptr_mode) {
    DLAF_CUDA_CALL(cudaSetDevice(stream_pool_.getDevice()));

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

  StreamHandle getNextHandle() {
    cudaStream_t stream = stream_pool_.getNextStream();
    cublasHandle_t handle = handles_[hpx::get_worker_thread_num()];
    DLAF_CUBLAS_CALL(cublasSetStream(handle, stream));
    DLAF_CUBLAS_CALL(cublasSetPointerMode(handle, ptr_mode_));
    return StreamHandle(stream, handle);
  }

  int getDevice() {
    return stream_pool_.getDevice();
  }

  cuda::StreamPool getStreamPool() {
    return stream_pool_;
  }
};
}

// Helper class with a reference counted CUBLAS handles and a reference to a
// StreamPool. Allows access to RAII handles (StreamHandle). Ensures
// that the correct device is set.
class HandlePool {
  std::shared_ptr<internal::HandlePoolImpl> handles_ptr_;

public:
  HandlePool(cuda::StreamPool stream_pool, cublasPointerMode_t ptr_mode = CUBLAS_POINTER_MODE_HOST)
      : handles_ptr_(std::make_shared<internal::HandlePoolImpl>(stream_pool, ptr_mode)) {}

  StreamHandle getNextHandle() {
    DLAF_ASSERT(bool(handles_ptr_), "");
    return handles_ptr_->getNextHandle();
  }

  int getDevice() {
    DLAF_ASSERT(bool(handles_ptr_), "");
    return handles_ptr_->getDevice();
  }

  cuda::StreamPool getStreamPool() {
    DLAF_ASSERT(bool(handles_ptr_), "");
    return handles_ptr_->getStreamPool();
  }

  bool operator==(HandlePool const& rhs) const noexcept {
    return handles_ptr_ == rhs.handles_ptr_;
  }

  bool operator!=(HandlePool const& rhs) const noexcept {
    return !(*this == rhs);
  }
};

/// An executor for CUBLAS functions. Each device has a single CUBLAS handle
/// associated to it. A CUBLAS function is defined as any function that takes a
/// CUBLAS handle as the first argument. The executor inserts a CUBLAS handle
/// into the argument list, i.e. a handle should not be provided at the
/// apply/async/dataflow invocation site.
class Executor : public cuda::Executor {
  using base = cuda::Executor;
  HandlePool handle_pool_;

public:
  Executor(HandlePool handle_pool) : base(handle_pool.getStreamPool()), handle_pool_(handle_pool) {}

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
    StreamHandle handle = handle_pool_.getNextHandle();
    auto r = hpx::invoke(std::forward<F>(f), handle.getHandle(), std::forward<Ts>(ts)...);
    hpx::future<void> fut = hpx::cuda::experimental::detail::get_future_with_event(handle.getStream());

    return fut.then(hpx::launch::sync,
                    [r = std::move(r)](hpx::future<void>&&) mutable { return std::move(r); });
  }

  template <class Frame, class F, class Futures>
  void dataflow_finalize(Frame&& frame, F&& f, Futures&& futures) {
    // Ensure the dataflow frame stays alive long enough.
    hpx::intrusive_ptr<typename std::remove_pointer<typename std::decay<Frame>::type>::type> frame_p(
        frame);

    StreamHandle handle = handle_pool_.getNextHandle();
    auto r = hpx::invoke_fused(std::forward<F>(f), hpx::tuple_cat(hpx::tie(handle.getHandle()),
                                                                  std::forward<Futures>(futures)));
    hpx::future<void> fut = hpx::cuda::experimental::detail::get_future_with_event(handle.getStream());

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
