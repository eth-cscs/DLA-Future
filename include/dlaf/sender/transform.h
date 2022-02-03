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

#include <pika/execution.hpp>
#include <pika/unwrap.hpp>
#include <pika/version.hpp>

#include "dlaf/init.h"
#include "dlaf/schedulers.h"
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

/// The Backend::MC specialization uses regular thread pool scheduler from pika.
template <>
struct Transform<Backend::MC> {
  template <typename S, typename F>
  static auto call(const Policy<Backend::MC> policy, S&& s, F&& f) {
    namespace ex = pika::execution::experimental;

    auto scheduler = getBackendScheduler<Backend::MC>(policy.priority());

    return ex::then(ex::transfer(std::forward<S>(s), std::move(scheduler)),
                    pika::unwrapping(std::forward<F>(f)));
  }
};

#ifdef DLAF_WITH_CUDA
template <>
struct Transform<Backend::GPU> {
  template <typename S, typename F>
  static auto call(const Policy<Backend::GPU> policy, S&& s, F&& f) {
    namespace ex = pika::execution::experimental;
    namespace cu = pika::cuda::experimental;

    auto scheduler = getBackendScheduler<Backend::GPU>(policy.priority());
    auto cuda_sender = ex::transfer(std::forward<S>(s), std::move(scheduler));
    auto f_unwrapping = pika::unwrapping(std::forward<F>(f));

    return cu::then_with_any_cuda(std::move(cuda_sender), std::move(f_unwrapping),
                                  CUBLAS_POINTER_MODE_HOST);
  }
};
#endif

/// Lazy transform. This does not submit the work and returns a sender.
template <Backend B, typename F, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
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
  pika::execution::experimental::start_detached(
      transformLift<B>(policy, std::forward<F>(f), std::forward<Ts>(ts)...));
}
}
}
