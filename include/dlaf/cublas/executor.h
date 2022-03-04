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

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <pika/execution.hpp>
#include <pika/functional.hpp>
#include <pika/future.hpp>
#include <pika/modules/async_cuda.hpp>
#include <pika/mutex.hpp>
#include <pika/tuple.hpp>
#include <pika/type_traits.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/cublas/error.h"
#include "dlaf/cublas/handle_pool.h"
#include "dlaf/cuda/error.h"
#include "dlaf/cuda/executor.h"

namespace dlaf {
namespace cublas {
namespace internal {

template <bool IsCallable, typename F, typename... Ts>
struct isAsyncCublasCallableImpl : std::false_type {
  struct dummy_type {};
  using return_type = dummy_type;
};

template <typename F, typename... Ts>
struct isAsyncCublasCallableImpl<true, F, Ts...> : std::true_type {
  using return_type = pika::future<typename pika::invoke_result<F, cublasHandle_t&, Ts...>::type>;
};

template <typename F, typename... Ts>
struct isAsyncCublasCallable
    : isAsyncCublasCallableImpl<pika::is_invocable_v<F, cublasHandle_t&, Ts...>, F, Ts...> {};

template <typename F, typename... Ts>
inline constexpr bool isAsyncCublasCallable_v = isAsyncCublasCallable<F, Ts...>::value;

template <typename F, typename Futures>
struct isDataflowCublasCallable
    : pika::is_invocable<pika::util::functional::invoke_fused, F,
                         decltype(pika::tuple_cat(pika::tie(std::declval<cublasHandle_t&>()),
                                                  std::declval<Futures>()))> {};
template <typename F, typename Futures>
inline constexpr bool isDataflowCublasCallable_v = isDataflowCublasCallable<F, Futures>::value;
}

/// An executor for cuBLAS functions. Uses handles and streams from the given
/// HandlePool and StreamPool. A cuBLAS function is defined as any function that
/// takes a cuBLAS handle as the first argument. The executor inserts a cuBLAS
/// handle into the argument list, i.e. a handle should not be provided at the
/// apply/async/dataflow invocation site.
class Executor : public cuda::Executor {
  using base = cuda::Executor;

protected:
  HandlePool handle_pool_;

public:
  Executor(cuda::StreamPool stream_pool, HandlePool handle_pool)
      : base(stream_pool), handle_pool_(handle_pool) {
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
  std::enable_if_t<internal::isAsyncCublasCallable_v<F, Ts...>,
                   typename internal::isAsyncCublasCallable<F, Ts...>::return_type>
  async_execute(F&& f, Ts&&... ts) {
    cudaStream_t stream = stream_pool_.getNextStream();
    cublasHandle_t handle = handle_pool_.getNextHandle(stream);
    auto r = pika::invoke(std::forward<F>(f), handle, std::forward<Ts>(ts)...);
    pika::future<void> fut = pika::cuda::experimental::detail::get_future_with_event(stream);

    // The handle and stream pools are captured by value to ensure that the
    // streams live at least until the event has completed.
    return fut.then(pika::launch::sync,
                    [r = std::move(r), stream_pool = stream_pool_,
                     handle_pool = handle_pool_](pika::future<void>&&) mutable { return std::move(r); });
  }

  template <class Frame, class F, class Futures>
  std::enable_if_t<internal::isDataflowCublasCallable_v<F, Futures>> dataflow_finalize(
      Frame&& frame, F&& f, Futures&& futures) {
    // Ensure the dataflow frame stays alive long enough.
    pika::intrusive_ptr<typename std::remove_pointer<typename std::decay<Frame>::type>::type> frame_p(
        frame);

    cudaStream_t stream = stream_pool_.getNextStream();
    cublasHandle_t handle = handle_pool_.getNextHandle(stream);
    auto r = pika::invoke_fused(std::forward<F>(f),
                                pika::tuple_cat(pika::tie(handle), std::forward<Futures>(futures)));
    pika::future<void> fut = pika::cuda::experimental::detail::get_future_with_event(stream);

    // The handle and stream pools are captured by value to ensure that the
    // streams live at least until the event has completed.
    fut.then(pika::launch::sync, [r = std::move(r), frame_p = std::move(frame_p),
                                  stream_pool = stream_pool_, handle_pool = handle_pool_](
                                     pika::future<void>&&) mutable { frame_p->set_data(std::move(r)); });
  }
};
}
}

namespace pika {
namespace parallel {
namespace execution {
template <>
struct is_two_way_executor<dlaf::cublas::Executor> : std::true_type {};
}
}
}

#endif
