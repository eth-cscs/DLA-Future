//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <pika/execution.hpp>

#include <dlaf/common/consume_rvalues.h>
#include <dlaf/common/unwrap.h>
#include <dlaf/init.h>
#include <dlaf/schedulers.h>
#include <dlaf/sender/policy.h>
#include <dlaf/sender/typelist.h>
#include <dlaf/sender/when_all_lift.h>
#include <dlaf/types.h>

#ifdef DLAF_WITH_GPU
#include <pika/cuda.hpp>

#include <dlaf/gpu/blas/api.h>
#include <dlaf/gpu/lapack/api.h>
#endif

#include <type_traits>

namespace dlaf {
namespace internal {

// Both rocblas and rocsolver functions are called with a rocblas_handle. This
// tag is here to disambiguate the call.
enum class TransformDispatchType { Plain, Blas, Lapack };

// The following are DLA-Future-specific transforms, with some helper variations
// for convenience and to approximate the behaviour of dataflow. Unlike
// execution::then, the transforms below insert additional arguments for the GPU
// backend (i.e. CUDA streams or cuBLAS/cuSOLVER handles). Additionally, the
// selection of which context to run on is hidden behind the Policy which also
// allows choosing the priority.
//
// At its core, transform is a convenience wrapper around
// sender | transfer(with_priority(scheduler, priority)) | then(ConsumeRvalues(unwrapping(f))).

/// Lazy transform. This does not submit the work and returns a sender.
template <TransformDispatchType Tag = TransformDispatchType::Plain, Backend B = Backend::MC,
          typename F = void, typename Sender = void,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
[[nodiscard]] decltype(auto) transform(const Policy<B> policy, F&& f, Sender&& sender) {
  using pika::execution::experimental::then;
  using pika::execution::experimental::transfer;

  auto scheduler = getBackendScheduler<B>(policy.priority());
  auto transfer_sender = transfer(std::forward<Sender>(sender), std::move(scheduler));

  using dlaf::common::internal::ConsumeRvalues;
  using dlaf::common::internal::Unwrapping;

  if constexpr (B == Backend::MC) {
    return then(std::move(transfer_sender), ConsumeRvalues{Unwrapping{std::forward<F>(f)}});
  }
  else if constexpr (B == Backend::GPU) {
#if defined(DLAF_WITH_GPU)
    using pika::cuda::experimental::then_with_cublas;
    using pika::cuda::experimental::then_with_cusolver;
    using pika::cuda::experimental::then_with_stream;

    if constexpr (Tag == TransformDispatchType::Plain) {
      return then_with_stream(std::move(transfer_sender),
                              ConsumeRvalues{Unwrapping{std::forward<F>(f)}});
    }
    else if constexpr (Tag == TransformDispatchType::Blas) {
      return then_with_cublas(std::move(transfer_sender),
                              ConsumeRvalues{Unwrapping{std::forward<F>(f)}}, CUBLAS_POINTER_MODE_HOST);
    }
    else if constexpr (Tag == TransformDispatchType::Lapack) {
      return then_with_cusolver(std::move(transfer_sender),
                                ConsumeRvalues{Unwrapping{std::forward<F>(f)}});
    }
    else {
      DLAF_STATIC_FAIL(
          Sender,
          "Attempting to use transform with a GPU policy, but f is not invocable with a CUDA stream as the last argument or cuBLAS/cuSOLVER handle as the first argument.");
    }
#else
    DLAF_STATIC_FAIL(Sender, "Attempting to use transform with Backend::GPU but it is disabled");
#endif
  }
  else {
    DLAF_STATIC_FAIL(Sender, "Unknown backend given to transform");
  }
}

/// Fire-and-forget transform. This submits the work and returns void.
template <TransformDispatchType Tag = TransformDispatchType::Plain, Backend B = Backend::MC,
          typename F = void, typename Sender = void,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
void transformDetach(const Policy<B> policy, F&& f, Sender&& sender) {
  pika::execution::experimental::start_detached(transform<Tag>(policy, std::forward<F>(f),
                                                               std::forward<Sender>(sender)));
}

/// Lazy transform. This does not submit the work and returns a sender. First
/// lifts non-senders into senders using just, and then calls transform with a
/// when_all sender of the lifted senders.
template <TransformDispatchType Tag, Backend B, typename F, typename... Ts>
[[nodiscard]] decltype(auto) transformLift(const Policy<B> policy, F&& f, Ts&&... ts) {
  return transform<Tag>(policy, std::forward<F>(f), internal::whenAllLift(std::forward<Ts>(ts)...));
}

/// Fire-and-forget transform. This submits the work and returns void. First
/// lifts non-senders into senders using just, and then calls transform with a
/// when_all sender of the lifted senders.
template <TransformDispatchType Tag = TransformDispatchType::Plain, Backend B = Backend::MC,
          typename F = void, typename... Ts>
void transformLiftDetach(const Policy<B> policy, F&& f, Ts&&... ts) {
  pika::execution::experimental::start_detached(transformLift<Tag>(policy, std::forward<F>(f),
                                                                   std::forward<Ts>(ts)...));
}

template <TransformDispatchType Tag, Backend B, typename F>
struct PartialTransformBase {
  const Policy<B> policy_;
  std::decay_t<F> f_;
};

/// A partially applied transform, with the policy and callable object given,
/// but the predecessor sender missing. The predecessor sender is applied when
/// calling the operator| overload.
template <TransformDispatchType Tag, Backend B, typename F>
class PartialTransform : private PartialTransformBase<Tag, B, F> {
public:
  template <typename F_>
  PartialTransform(const Policy<B> policy, F_&& f)
      : PartialTransformBase<Tag, B, F>{policy, std::forward<F_>(f)} {}
  PartialTransform(PartialTransform&&) = default;
  PartialTransform(const PartialTransform&) = default;
  PartialTransform& operator=(PartialTransform&&) = default;
  PartialTransform& operator=(const PartialTransform&) = default;

  template <typename Sender>
  friend auto operator|(Sender&& sender, const PartialTransform pa) {
    return transform<Tag, B>(pa.policy_, std::move(pa.f_), std::forward<Sender>(sender));
  }
};

template <TransformDispatchType Tag, Backend B, typename F>
auto makePartialTransform(const Policy<B> policy, F&& f) {
  return PartialTransform<Tag, B, std::decay_t<F>>{policy, std::forward<F>(f)};
}

/// A partially applied transformDetach, with the policy and callable object
/// given, but the predecessor sender missing. The predecessor sender is applied
/// when calling the operator| overload.
template <TransformDispatchType Tag, Backend B, typename F>
class PartialTransformDetach : private PartialTransformBase<Tag, B, F> {
public:
  template <typename F_>
  PartialTransformDetach(const Policy<B> policy, F_&& f)
      : PartialTransformBase<Tag, B, F>{policy, std::forward<F_>(f)} {}
  PartialTransformDetach(PartialTransformDetach&&) = default;
  PartialTransformDetach(const PartialTransformDetach&) = default;
  PartialTransformDetach& operator=(PartialTransformDetach&&) = default;
  PartialTransformDetach& operator=(const PartialTransformDetach&) = default;

  template <typename Sender>
  friend auto operator|(Sender&& sender, const PartialTransformDetach pa) {
    return pika::execution::experimental::start_detached(
        transform<Tag, B>(pa.policy_, std::move(pa.f_), std::forward<Sender>(sender)));
  }
};

template <TransformDispatchType Tag, Backend B, typename F>
auto makePartialTransformDetach(const Policy<B> policy, F&& f) {
  return PartialTransformDetach<Tag, B, std::decay_t<F>>{policy, std::forward<F>(f)};
}

/// \overload transform
///
/// This overload partially applies the transform for later use with operator|
/// with a sender on the left-hand side.
template <TransformDispatchType Tag = TransformDispatchType::Plain, Backend B = Backend::MC,
          typename F = void>
[[nodiscard]] decltype(auto) transform(const Policy<B> policy, F&& f) {
  return makePartialTransform<Tag>(policy, std::forward<F>(f));
}

/// \overload transformDetach
///
/// This overload partially applies transformDetach for later use with operator|
/// with a sender on the left-hand side.
template <TransformDispatchType Tag = TransformDispatchType::Plain, Backend B = Backend::MC,
          typename F = void>
[[nodiscard]] decltype(auto) transformDetach(const Policy<B> policy, F&& f) {
  return makePartialTransformDetach<Tag>(policy, std::forward<F>(f));
}
}
}
