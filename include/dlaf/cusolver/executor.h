//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
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

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <hpx/local/execution.hpp>
#include <hpx/local/functional.hpp>
#include <hpx/local/future.hpp>
#include <hpx/local/mutex.hpp>
#include <hpx/local/tuple.hpp>
#include <hpx/local/type_traits.hpp>
#include <hpx/modules/async_cuda.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/cublas/executor.h"
#include "dlaf/cublas/handle_pool.h"
#include "dlaf/cuda/error.h"
#include "dlaf/cusolver/error.h"
#include "dlaf/cusolver/handle_pool.h"

namespace dlaf {
namespace cusolver {
namespace internal {
template <bool IsCallable, typename F, typename... Ts>
struct isAsyncCusolverCallableImpl : std::false_type {
  struct dummy_type {};
  using return_type = dummy_type;
};

template <typename F, typename... Ts>
struct isAsyncCusolverCallableImpl<true, F, Ts...> : std::true_type {
  using return_type = hpx::future<typename hpx::invoke_result<F, cusolverDnHandle_t&, Ts...>::type>;
};

template <typename F, typename... Ts>
struct isAsyncCusolverCallable
    : isAsyncCusolverCallableImpl<hpx::is_invocable_v<F, cusolverDnHandle_t&, Ts...>, F, Ts...> {};

template <typename F, typename Futures>
struct isDataflowCusolverCallable
    : hpx::is_invocable<hpx::util::functional::invoke_fused, F,
                        decltype(hpx::tuple_cat(hpx::tie(std::declval<cusolverDnHandle_t&>()),
                                                std::declval<Futures>()))> {};

template <typename F, typename Futures>
inline constexpr bool isDataflowCusolverCallable_v = isDataflowCusolverCallable<F, Futures>::value;
}

/// An executor for cuSOLVER functions. Uses handles and streams from the given
/// HandlePool and StreamPool. A cuSOLVER function is defined as any function
/// that takes a cuSOLVER handle as the first argument. The executor inserts a
/// cuSOLVER handle into the argument list, i.e. a handle should not be
/// provided at the apply/async/dataflow invocation site.
class Executor : public cublas::Executor {
  using base = cublas::Executor;
  HandlePool handle_pool_;

public:
  Executor(cuda::StreamPool stream_pool, cublas::HandlePool cublas_handle_pool, HandlePool handle_pool)
      : base(stream_pool, cublas_handle_pool), handle_pool_(handle_pool) {
    DLAF_ASSERT(base::handle_pool_.getDevice() == handle_pool_.getDevice(),
                base::handle_pool_.getDevice(), handle_pool_.getDevice());
    DLAF_ASSERT(stream_pool_.getDevice() == handle_pool_.getDevice(), stream_pool_.getDevice(),
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
  std::enable_if_t<internal::isAsyncCusolverCallable_v<F, Ts...>,
                   typename internal::isAsyncCusolverCallable<F, Ts...>::return_type>
  async_execute(F&& f, Ts&&... ts) {
    cudaStream_t stream = stream_pool_.getNextStream();
    cusolverDnHandle_t handle = handle_pool_.getNextHandle(stream);
    auto r = hpx::invoke(std::forward<F>(f), handle, std::forward<Ts>(ts)...);
    hpx::future<void> fut = hpx::cuda::experimental::detail::get_future_with_event(stream);

    // The handle and stream pools are captured by value to ensure that the
    // streams live at least until the event has completed.
    return fut.then(hpx::launch::sync,
                    [r = std::move(r), stream_pool = stream_pool_,
                     handle_pool = handle_pool_](hpx::future<void>&&) mutable { return std::move(r); });
  }

  template <class Frame, class F, class Futures>
  std::enable_if_t<internal::isDataflowCusolverCallable_v<F, Futures>> dataflow_finalize(
      Frame&& frame, F&& f, Futures&& futures) {
    // Ensure the dataflow frame stays alive long enough.
    hpx::intrusive_ptr<typename std::remove_pointer<typename std::decay<Frame>::type>::type> frame_p(
        frame);

    cudaStream_t stream = stream_pool_.getNextStream();
    cusolverDnHandle_t handle = handle_pool_.getNextHandle(stream);
    auto r = hpx::invoke_fused(std::forward<F>(f),
                               hpx::tuple_cat(hpx::tie(handle), std::forward<Futures>(futures)));
    hpx::future<void> fut = hpx::cuda::experimental::detail::get_future_with_event(stream);

    // The handle and stream pools are captured by value to ensure that the
    // streams live at least until the event has completed.
    fut.then(hpx::launch::sync, [r = std::move(r), frame_p = std::move(frame_p),
                                 stream_pool = stream_pool_, handle_pool = handle_pool_](
                                    hpx::future<void>&&) mutable { frame_p->set_data(std::move(r)); });
  }

  template <typename F, typename... Ts>
  std::enable_if_t<cublas::internal::isAsyncCublasCallable_v<F, Ts...>,
                   typename cublas::internal::isAsyncCublasCallable<F, Ts...>::return_type>
  async_execute(F&& f, Ts&&... ts) {
    return base::async_execute(std::forward<F>(f), std::forward<Ts>(ts)...);
  }

  template <class Frame, class F, class Futures>
  std::enable_if_t<cublas::internal::isDataflowCublasCallable_v<F, Futures>> dataflow_finalize(
      Frame&& frame, F&& f, Futures&& futures) {
    base::dataflow_finalize(std::forward<Frame>(frame), std::forward<F>(f),
                            std::forward<Futures>(futures));
  }
};
}
}

namespace hpx {
namespace parallel {
namespace execution {
template <>
struct is_two_way_executor<dlaf::cusolver::Executor> : std::true_type {};
}
}
}

#endif
