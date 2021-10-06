//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2020-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <hpx/local/execution.hpp>
#include <hpx/local/unwrap.hpp>

#include "dlaf/init.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/typelist.h"
#include "dlaf/sender/when_all_lift.h"
#include "dlaf/types.h"

#ifdef DLAF_WITH_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "dlaf/cublas/handle_pool.h"
#include "dlaf/cuda/stream_pool.h"
#include "dlaf/cusolver/handle_pool.h"
#endif

namespace dlaf {
namespace internal {
/// DLAF-specific transform, templated on a backend. This, together with
/// when_all, takes the place of dataflow(executor, ...) for futures.
template <Backend B>
struct Transform;

/// The Backend::MC specialization uses regular thread pool scheduler from HPX.
template <>
struct Transform<Backend::MC> {
  template <typename S, typename F>
  static auto call(const Policy<Backend::MC> policy, S&& s, F&& f) {
    namespace ex = hpx::execution::experimental;
    return ex::transform(ex::on(std::forward<S>(s),
                                ex::with_priority(ex::thread_pool_scheduler{}, policy.priority())),
                         hpx::unwrapping(std::forward<F>(f)));
  }
};

#ifdef DLAF_WITH_CUDA
/// The Backend::GPU specialization uses a custom sender. The custom sender,
/// when connected to a receiver, chooses an approprate stream or handle pool
/// depending on what the callable accepts, calls the given callable with an
/// element from a pool, and signals the receiver when the operation is ready
/// (notified using a CUDA event).
template <>
struct Transform<Backend::GPU> {
  template <typename S, typename F>
  struct GPUTransformSender {
    cuda::StreamPool stream_pool;
    cublas::HandlePool cublas_handle_pool;
    cusolver::HandlePool cusolver_handle_pool;
    std::decay_t<S> s;
    std::decay_t<F> f;

    template <typename G, typename... Us>
    static auto call_helper(cudaStream_t stream, cublasHandle_t cublas_handle,
                            cusolverDnHandle_t cusolver_handle, G&& g, Us&... us) {
      using unwrapping_function_type = decltype(hpx::unwrapping(std::forward<G>(g)));
      constexpr bool is_cuda_stream_invocable =
          std::is_invocable_v<unwrapping_function_type, Us&..., cudaStream_t>;
      constexpr bool is_cublas_handle_invocable =
          std::is_invocable_v<unwrapping_function_type, cublasHandle_t, Us&...>;
      constexpr bool is_cusolver_handle_invocable =
          std::is_invocable_v<unwrapping_function_type, cusolverDnHandle_t, Us&...>;
      static_assert(is_cuda_stream_invocable || is_cublas_handle_invocable ||
                        is_cusolver_handle_invocable,
                    "function passed to transform<GPU> must be invocable with a cublasStream_t as the "
                    "last argument or a cublasHandle_t/cusolverDnHandle_t as the first argument");

      if constexpr (is_cuda_stream_invocable) {
        (void) cublas_handle;
        (void) cusolver_handle;
        return std::invoke(hpx::unwrapping(std::forward<G>(g)), us..., stream);
      }
      else if constexpr (is_cublas_handle_invocable) {
        (void) cusolver_handle;
        (void) stream;
        return std::invoke(hpx::unwrapping(std::forward<G>(g)), cublas_handle, us...);
      }
      else if constexpr (is_cusolver_handle_invocable) {
        (void) cublas_handle;
        (void) stream;
        return std::invoke(hpx::unwrapping(std::forward<G>(g)), cusolver_handle, us...);
      }
    }

    template <typename Tuple>
    struct invoke_result_helper;

    template <template <typename...> class Tuple, typename... Ts>
    struct invoke_result_helper<Tuple<Ts...>> {
      using result_type = decltype(
          call_helper(std::declval<cudaStream_t&>(), std::declval<cublasHandle_t&>(),
                      std::declval<cusolverDnHandle_t&>(), std::declval<F>(), std::declval<Ts&>()...));
      using type =
          typename std::conditional<std::is_void<result_type>::value, Tuple<>, Tuple<result_type>>::type;
    };

    template <template <typename...> class Tuple, template <typename...> class Variant>
    using value_types = dlaf::internal::UniquePackT<dlaf::internal::TransformPackT<
        typename hpx::execution::experimental::sender_traits<S>::template value_types<Tuple, Variant>,
        invoke_result_helper>>;

    template <template <typename...> class Variant>
    using error_types = dlaf::internal::UniquePackT<dlaf::internal::PrependPackT<
        typename hpx::execution::experimental::sender_traits<S>::template error_types<Variant>,
        std::exception_ptr>>;

    static constexpr bool sends_done = false;

    template <typename R>
    struct GPUTransformReceiver {
      cuda::StreamPool stream_pool;
      cublas::HandlePool cublas_handle_pool;
      cusolver::HandlePool cusolver_handle_pool;
      std::decay_t<R> r;
      std::decay_t<F> f;

      template <typename E>
      void set_error(E&& e) && noexcept {
        hpx::execution::experimental::set_error(std::move(r), std::forward<E>(e));
      }

      void set_done() && noexcept {
        hpx::execution::experimental::set_done(std::move(r));
      }

      template <typename... Ts>
      void set_value(Ts&&... ts) noexcept {
        try {
          cudaStream_t stream = stream_pool.getNextStream();
          cublasHandle_t cublas_handle = cublas_handle_pool.getNextHandle(stream);
          cusolverDnHandle_t cusolver_handle = cusolver_handle_pool.getNextHandle(stream);

          // NOTE: We do not forward ts because we keep the pack alive longer in
          // the continuation.
          if constexpr (std::is_void_v<decltype(
                            call_helper(stream, cublas_handle, cusolver_handle, std::move(f), ts...))>) {
            call_helper(stream, cublas_handle, cusolver_handle, std::move(f), ts...);
            hpx::cuda::experimental::detail::add_event_callback(
                [r = std::move(r),
                 keep_alive =
                     std::make_tuple(std::forward<Ts>(ts)..., std::move(stream_pool),
                                     std::move(cublas_handle_pool),
                                     std::move(cusolver_handle_pool))](cudaError_t status) mutable {
                  DLAF_CUDA_CALL(status);
                  hpx::execution::experimental::set_value(std::move(r));
                },
                stream);
          }
          else {
            auto res = call_helper(stream, cublas_handle, cusolver_handle, std::move(f), ts...);
            hpx::cuda::experimental::detail::add_event_callback(
                [r = std::move(r), res = std::move(res),
                 keep_alive =
                     std::make_tuple(std::forward<Ts>(ts)..., std::move(stream_pool),
                                     std::move(cublas_handle_pool),
                                     std::move(cusolver_handle_pool))](cudaError_t status) mutable {
                  DLAF_CUDA_CALL(status);
                  hpx::execution::experimental::set_value(std::move(r), std::move(res));
                },
                stream);
          }
        }
        catch (...) {
          hpx::execution::experimental::set_error(std::move(r), std::current_exception());
        }
      }
    };

    template <typename R>
    auto connect(R&& r) && {
      return hpx::execution::experimental::connect(std::move(s),
                                                   GPUTransformReceiver<R>{stream_pool,
                                                                           cublas_handle_pool,
                                                                           cusolver_handle_pool,
                                                                           std::forward<R>(r),
                                                                           std::move(f)});
    }
  };

  template <typename S, typename F>
  static auto call(const Policy<Backend::GPU> policy, S&& s, F&& f) {
    return GPUTransformSender<S, F>{policy.priority() >= hpx::threads::thread_priority::high
                                        ? getHpCudaStreamPool()
                                        : getNpCudaStreamPool(),
                                    getCublasHandlePool(), getCusolverHandlePool(), std::forward<S>(s),
                                    std::forward<F>(f)};
  }
};
#endif

/// Lazy transform. This does not submit the work and returns a sender.
template <Backend B, typename F, typename Sender,
          typename = std::enable_if_t<hpx::execution::experimental::is_sender_v<Sender>>>
[[nodiscard]] decltype(auto) transform(const Policy<B> policy, F&& f, Sender&& sender) {
  return internal::Transform<B>::call(policy, std::forward<Sender>(sender), std::forward<F>(f));
}

/// Lazy transform. This does not submit the work and returns a sender. First
/// lifts non-senders into senders using just, and then calls transform with a
/// when_all sender of the lifted senders.
template <Backend B, typename F, typename... Ts>
[[nodiscard]] decltype(auto) transformLift(const Policy<B> policy, F&& f, Ts&&... ts) {
  return internal::Transform<B>::call(policy, internal::whenAllLift(std::forward<Ts>(ts)...),
                                      std::forward<F>(f));
}

/// Fire-and-forget transform. This submits the work and returns void. First
/// lifts non-senders into senders using just, and then calls transform with a
/// when_all sender of the lifted senders.
template <Backend B, typename F, typename... Ts>
void transformLiftDetach(const Policy<B> policy, F&& f, Ts&&... ts) {
  hpx::execution::experimental::detach(
      transformLift<B>(policy, std::forward<F>(f), std::forward<Ts>(ts)...));
}
}
}
