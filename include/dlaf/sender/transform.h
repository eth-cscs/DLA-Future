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

#include <hpx/local/execution.hpp>
#include <hpx/local/unwrap.hpp>
#include <hpx/version.hpp>

#include "dlaf/init.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/traits.h"
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
    return ex::then(ex::on(std::forward<S>(s),
                           ex::with_priority(ex::thread_pool_scheduler{}, policy.priority())),
                    hpx::unwrapping(std::forward<F>(f)));
  }
};

#ifdef DLAF_WITH_CUDA
/// The Backend::GPU specialization uses a custom sender. The custom sender,
/// when connected to a receiver, chooses an appropriate stream or handle pool
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

    using unwrapping_function_type = decltype(hpx::unwrapping(std::declval<std::decay_t<F>>()));
    template <typename... Ts>
    static constexpr bool is_cuda_stream_invocable =
        std::is_invocable_v<unwrapping_function_type, std::decay_t<Ts>&..., cudaStream_t>;
    template <typename... Ts>
    static constexpr bool is_cublas_handle_invocable =
        std::is_invocable_v<unwrapping_function_type, cublasHandle_t, std::decay_t<Ts>&...>;
    template <typename... Ts>
    static constexpr bool is_cusolver_handle_invocable =
        std::is_invocable_v<unwrapping_function_type, cusolverDnHandle_t, std::decay_t<Ts>&...>;
    template <typename... Ts>
    static constexpr bool is_gpu_invocable =
        is_cuda_stream_invocable<Ts...> || is_cublas_handle_invocable<Ts...> ||
        is_cusolver_handle_invocable<Ts...>;

    template <typename G, typename... Us>
    static auto call_helper(cudaStream_t stream, cublasHandle_t cublas_handle,
                            cusolverDnHandle_t cusolver_handle, G&& g, Us&... us) {
      static_assert(is_gpu_invocable<Us...>,
                    "function passed to transform<GPU> must be invocable with a cudaStream_t as the"
                    "last argument or a cublasHandle_t/cusolverDnHandle_t as the first argument");

      if constexpr (is_cuda_stream_invocable<Us...>) {
        (void) cublas_handle;
        (void) cusolver_handle;
        return std::invoke(hpx::unwrapping(std::forward<G>(g)), us..., stream);
      }
      else if constexpr (is_cublas_handle_invocable<Us...>) {
        (void) cusolver_handle;
        (void) stream;
        return std::invoke(hpx::unwrapping(std::forward<G>(g)), cublas_handle, us...);
      }
      else if constexpr (is_cusolver_handle_invocable<Us...>) {
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
          typename std::conditional<std::is_void_v<result_type>, Tuple<>, Tuple<result_type>>::type;
    };

    template <template <typename...> class Tuple, template <typename...> class Variant>
    using value_types = dlaf::internal::UniquePack_t<dlaf::internal::TransformPack_t<
        typename hpx::execution::experimental::sender_traits<S>::template value_types<Tuple, Variant>,
        invoke_result_helper>>;

    template <template <typename...> class Variant>
    using error_types = dlaf::internal::UniquePack_t<dlaf::internal::PrependPack_t<
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
      friend void tag_invoke(hpx::execution::experimental::set_error_t, GPUTransformReceiver&& r,
                             E&& e) noexcept {
        hpx::execution::experimental::set_error(std::move(r.r), std::forward<E>(e));
      }

      friend void tag_invoke(hpx::execution::experimental::set_done_t,
                             GPUTransformReceiver&& r) noexcept {
        hpx::execution::experimental::set_done(std::move(r));
      }

      template <typename... Ts, typename Enable = std::enable_if_t<is_gpu_invocable<Ts...>>>
      friend auto tag_invoke(hpx::execution::experimental::set_value_t, GPUTransformReceiver&& r,
                             Ts&&... ts) {
        try {
          cudaStream_t stream = r.stream_pool.getNextStream();
          cublasHandle_t cublas_handle = r.cublas_handle_pool.getNextHandle(stream);
          cusolverDnHandle_t cusolver_handle = r.cusolver_handle_pool.getNextHandle(stream);

          // NOTE: We do not forward ts because we keep the pack alive longer in
          // the continuation.
          if constexpr (std::is_void_v<decltype(call_helper(stream, cublas_handle, cusolver_handle,
                                                            std::move(r.f), ts...))>) {
            call_helper(stream, cublas_handle, cusolver_handle, std::move(r.f), ts...);
            hpx::cuda::experimental::detail::add_event_callback(
                [r = std::move(r.r),
                 keep_alive =
                     std::make_tuple(std::forward<Ts>(ts)..., std::move(r.stream_pool),
                                     std::move(r.cublas_handle_pool),
                                     std::move(r.cusolver_handle_pool))](cudaError_t status) mutable {
                  DLAF_CUDA_CALL(status);
                  hpx::execution::experimental::set_value(std::move(r));
                },
                stream);
          }
          else {
            auto res = call_helper(stream, cublas_handle, cusolver_handle, std::move(r.f), ts...);
            hpx::cuda::experimental::detail::add_event_callback(
                [r = std::move(r.r), res = std::move(res),
                 keep_alive =
                     std::make_tuple(std::forward<Ts>(ts)..., std::move(r.stream_pool),
                                     std::move(r.cublas_handle_pool),
                                     std::move(r.cusolver_handle_pool))](cudaError_t status) mutable {
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
    friend auto tag_invoke(hpx::execution::experimental::connect_t, GPUTransformSender&& s, R&& r) {
      return hpx::execution::experimental::connect(std::move(s.s),
                                                   GPUTransformReceiver<R>{std::move(s.stream_pool),
                                                                           std::move(
                                                                               s.cublas_handle_pool),
                                                                           std::move(
                                                                               s.cusolver_handle_pool),
                                                                           std::forward<R>(r),
                                                                           std::move(s.f)});
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
  hpx::execution::experimental::start_detached(
      transformLift<B>(policy, std::forward<F>(f), std::forward<Ts>(ts)...));
}
}
}
