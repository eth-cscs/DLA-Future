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
#include "dlaf/cuda/event.h"
#include "dlaf/cuda/executor.h"

namespace dlaf {
namespace cublas {
namespace internal {

// Helper class for initializing and destroying a CUBLAS handle.
struct CublasHandle {
  cublasHandle_t handle_;

  CublasHandle(int device) noexcept {
    DLAF_CUDA_CALL(cudaSetDevice(device));
    DLAF_CUBLAS_CALL(cublasCreate(&handle_));
  }

  ~CublasHandle() {
    // This implicitly calls `cublasDeviceSynchronize()` [1].
    //
    // [1]: cuBLAS, section 2.4 cuBLAS Helper Function Reference
    DLAF_CUBLAS_CALL(cublasDestroy(handle_));
  }
};

// Helper class which holds a CUBLAS handle, protected by a lock. This class is
// intended to be used as an RAII guard. The usage of the handle is protected
// as long as an instance is alive. It also provides a means to get an event
// corresponding to the stream of the handle. Instances of this object can not
// be shared among threads. The lifetime of an instance must be no longer than
// the HandlePool from which an instance was taken.
//
// NOTE: This currently relies on the lock of LockedStream.
class LockedHandle {
  cuda::internal::LockedStream locked_stream_;
  cublasHandle_t handle_;

public:
  LockedHandle(cuda::internal::LockedStream locked_stream, cublasHandle_t handle)
      : locked_stream_(std::move(locked_stream)), handle_(handle) {}

  cublasHandle_t& getHandle() {
    return handle_;
  }

  cudaStream_t& getStream() {
    return locked_stream_.getStream();
  }

  cuda::Event getEvent() {
    return locked_stream_.getEvent();
  }

  // TODO: Do we need variants here?
  // decltype(auto) getFutureCallback() {}
  // decltype(auto) getFutureEventSchedulerPolling() {}
  // decltype(auto) getFutureEventYieldPolling() {}

  void unlock() {
    locked_stream_.unlock();
  }
};

// Helper class with a reference counted CUBLAS handle and a reference to a
// StreamPool. Allows access to RAII locked handles (LockedHandle). Ensures
// that the correct device is set.
//
// NOTE: This currently only holds a single CUBLAS handle, which uses streams
// from the StreamPool.
//
// NOTE: This is only intended for use in the CUBLAS executor below. A
// reference to the StreamPool is enough since its lifetime will be the same as
// the base class, dlaf::cuda::Executor.
class HandlePool {
  cuda::internal::StreamPool stream_pool_;
  cublasPointerMode_t ptr_mode_;
  std::shared_ptr<CublasHandle> handle_ptr_;

public:
  HandlePool(cuda::internal::StreamPool stream_pool, cublasPointerMode_t ptr_mode, int device)
      : stream_pool_(stream_pool), ptr_mode_(ptr_mode),
        handle_ptr_(std::make_shared<CublasHandle>(device)) {}

  LockedHandle getNextHandle() {
    cuda::internal::LockedStream locked_stream = stream_pool_.getNextStream();
    DLAF_CUBLAS_CALL(cublasSetStream(handle_ptr_->handle_, locked_stream.getStream()));
    DLAF_CUBLAS_CALL(cublasSetPointerMode(handle_ptr_->handle_, ptr_mode_));
    return LockedHandle(std::move(locked_stream), handle_ptr_.get()->handle_);
  }

  bool operator==(HandlePool const& rhs) const noexcept {
    return stream_pool_ == rhs.stream_pool_ && ptr_mode_ == rhs.ptr_mode_ &&
           handle_ptr_ == rhs.handle_ptr_;
  }

  bool operator!=(HandlePool const& rhs) const noexcept {
    return !(*this == rhs);
  }
};
}

/// An executor for CUBLAS functions. Each device has a single CUBLAS handle
/// associated to it. A CUBLAS function is defined as any function that takes a
/// CUBLAS handle as the first argument. The executor inserts a CUBLAS handle
/// into the argument list, i.e. a handle should not be provided at the
/// apply/async/dataflow invocation site.
class Executor : public cuda::Executor {
  using base = cuda::Executor;
  internal::HandlePool handle_pool_;

public:
  Executor(int device, int num_streams,
           hpx::threads::thread_priority priority = hpx::threads::thread_priority_normal,
           cublasPointerMode_t ptr_mode = CUBLAS_POINTER_MODE_HOST)
      : base(device, num_streams, priority), handle_pool_(base::stream_pool_, ptr_mode, device) {}

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
    internal::LockedHandle locked_handle = handle_pool_.getNextHandle();
    auto r = hpx::invoke(std::forward<F>(f), locked_handle.getHandle(), std::forward<Ts>(ts)...);
    hpx::future<void> fut =
        hpx::cuda::experimental::detail::get_future_with_event(locked_handle.getStream());
    locked_handle.unlock();

    // TODO: If using get_future_with_callback, cudaFree may be called in the
    // callback. Do we care?
    return fut.then(hpx::launch::sync,
                    [r = std::move(r)](hpx::future<void>&&) mutable { return std::move(r); });
  }

  template <class Frame, class F, class Futures>
  void dataflow_finalize(Frame&& frame, F&& f, Futures&& futures) {
    // Ensure the dataflow frame stays alive long enough.
    hpx::intrusive_ptr<typename std::remove_pointer<typename std::decay<Frame>::type>::type> frame_p(
        frame);

    internal::LockedHandle locked_handle = handle_pool_.getNextHandle();
    auto r = hpx::invoke_fused(std::forward<F>(f), hpx::tuple_cat(hpx::tie(locked_handle.getHandle()),
                                                                  std::forward<Futures>(futures)));
    hpx::future<void> fut =
        hpx::cuda::experimental::detail::get_future_with_event(locked_handle.getStream());
    locked_handle.unlock();

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
