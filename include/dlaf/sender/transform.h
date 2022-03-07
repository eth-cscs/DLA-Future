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

#include "dlaf/init.h"
#include "dlaf/schedulers.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/typelist.h"
#include "dlaf/sender/when_all_lift.h"
#include "dlaf/types.h"

#ifdef DLAF_WITH_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <pika/modules/async_cuda.hpp>

#include "dlaf/cublas/handle_pool.h"
#include "dlaf/cuda/stream_pool.h"
#include "dlaf/cusolver/handle_pool.h"
#endif

namespace dlaf {
namespace internal {
// The following are DLA-Future-specific transforms, with some helper variations
// for convenience and to approximate the behaviour of dataflow. Unlike
// execution::then, the transforms below insert additional arguments for the GPU
// backend (i.e. CUDA streams or cuBLAS/cuSOLVER handles). Additionally, the
// selection of which context to run on is hidden behind the Policy which also
// allows choosing the priority.
//
// At its core, transform is a convenience wrapper around
// sender | transfer(with_priority(scheduler, priority)) | then(unwrapping(f)).

/// Lazy transform. This does not submit the work and returns a sender.
template <Backend B, typename F, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
[[nodiscard]] decltype(auto) transform(const Policy<B> policy, F&& f, Sender&& sender) {
  using pika::execution::experimental::then;
  using pika::execution::experimental::transfer;
  using pika::unwrapping;

  auto scheduler = getBackendScheduler<B>(policy.priority());
  auto transfer_sender = transfer(std::forward<Sender>(sender), std::move(scheduler));
  auto f_unwrapping = pika::unwrapping(std::forward<F>(f));

  if constexpr (B == Backend::MC) {
    return then(std::move(transfer_sender), std::move(f_unwrapping));
  }
  else if constexpr (B == Backend::GPU) {
#if defined(DLAF_WITH_CUDA)
    using pika::cuda::experimental::then_with_cublas;
    using pika::cuda::experimental::then_with_cusolver;
    using pika::cuda::experimental::then_with_stream;

    if constexpr (std::is_invocable_v<decltype(then_with_stream), decltype(std::move(transfer_sender)),
                                      decltype(std::move(f_unwrapping))>) {
      return then_with_stream(std::move(transfer_sender), std::move(f_unwrapping));
    }
    else if constexpr (std::is_invocable_v<decltype(then_with_cublas),
                                           decltype(std::move(transfer_sender)),
                                           decltype(std::move(f_unwrapping)), cublasPointerMode_t>) {
      return then_with_cublas(std::move(transfer_sender), std::move(f_unwrapping),
                              CUBLAS_POINTER_MODE_HOST);
    }
    else if constexpr (std::is_invocable_v<decltype(then_with_cusolver),
                                           decltype(std::move(transfer_sender)),
                                           decltype(std::move(f_unwrapping))>) {
      return then_with_cusolver(std::move(transfer_sender), std::move(f_unwrapping));
    }
    else {
      DLAF_STATIC_FAIL(
          F,
          "Attempting to use transform with a GPU policy, but f is not invocable with a CUDA stream as the last argument or cuBLAS/cuSOLVER handle as the first argument.");
    }
#else
    DLAF_STATIC_FAIL(F, "Attempting to use transform with Backend::GPU but it is disabled");
#endif
  }
  else {
    DLAF_STATIC_FAIL(F, "Unknown backend given to transform");
  }
}

/// Lazy transform. This does not submit the work and returns a sender. First
/// lifts non-senders into senders using just, and then calls transform with a
/// when_all sender of the lifted senders.
template <Backend B, typename F, typename... Ts>
[[nodiscard]] decltype(auto) transformLift(const Policy<B> policy, F&& f, Ts&&... ts) {
  return transform<B>(policy, std::forward<F>(f), whenAllLift(std::forward<Ts>(ts)...));
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
