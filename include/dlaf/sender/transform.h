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

#ifdef DLAF_WITH_GPU
#include "dlaf/gpu/api.h"
#include "dlaf/gpu/blas/api.h"
#include "dlaf/gpu/lapack/api.h"

#include <pika/cuda.hpp>
#endif

namespace dlaf {
namespace internal {

// hipBLAS functions take a handle of type hipblasHandle_t which is a typedef
// for a void pointer. Those functions can be therefore called with a
// rocblas_handle (handle used for rocsolver functions). This tag is here to
// disambiguate the call.
enum class TransformDispatchType { Plain, Blas, Lapack };

// The following are DLA-Future-specific transforms, with some helper variations
// for convenience and to approximate the behaviour of dataflow. Unlike
// execution::then, the transforms below insert additional arguments for the GPU
// backend (i.e. CUDA streams or cuBLAS/cuSOLVER handles). Additionally, the
// selection of which context to run on is hidden behind the Policy which also
// allows choosing the priority.
//
// At its core, transform is a convenience wrapper around
// sender | transfer(with_priority(scheduler, priority)) | then(unwrapping(f)).

template <typename T>
struct IsReferenceWrapper : std::false_type {};

template <typename U>
struct IsReferenceWrapper<std::reference_wrapper<U>> : std::true_type {};

template <typename T>
decltype(auto) getReferenceWrapper(T&& t) {
  if constexpr (IsReferenceWrapper<std::decay_t<T>>::value) {
    return t.get();
  }
  else {
    return std::forward<T>(t);
  }
}

template <typename F>
struct TransformCallHelper {
  std::decay_t<F> f;
  template <typename... Ts>
  auto operator()(Ts&&... ts) -> decltype(std::move(f)(getReferenceWrapper(std::forward<Ts>(ts))...)) {
    return std::move(f)(getReferenceWrapper(std::forward<Ts>(ts))...);
  }
};

template <typename F>
TransformCallHelper(F &&) -> TransformCallHelper<std::decay_t<F>>;

/// Lazy transform. This does not submit the work and returns a sender.
template <TransformDispatchType Tag = TransformDispatchType::Plain, bool Unwrap = true,
          Backend B = Backend::MC, typename F = void, typename Sender = void,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
[[nodiscard]] decltype(auto) transform(const Policy<B> policy, F&& f, Sender&& sender) {
  using pika::unwrapping;
  using pika::execution::experimental::then;
  using pika::execution::experimental::transfer;

  auto scheduler = getBackendScheduler<B>(policy.priority());
  auto transfer_sender = transfer(std::forward<Sender>(sender), std::move(scheduler));
  auto f_unwrapping = [&]() {
    if constexpr (Unwrap) {
      return pika::unwrapping(TransformCallHelper{std::forward<F>(f)});
    }
    else {
      return TransformCallHelper{std::forward<F>(f)};
    }
  }();

  if constexpr (B == Backend::MC) {
    return then(std::move(transfer_sender), std::move(f_unwrapping));
  }
  else if constexpr (B == Backend::GPU) {
#if defined(DLAF_WITH_GPU)
    using pika::cuda::experimental::then_with_cublas;
    using pika::cuda::experimental::then_with_cusolver;
    using pika::cuda::experimental::then_with_stream;

    if constexpr (Tag == TransformDispatchType::Plain) {
      return then_with_stream(std::move(transfer_sender), std::move(f_unwrapping));
    }
    else if constexpr (Tag == TransformDispatchType::Blas) {
      return then_with_cublas(std::move(transfer_sender), std::move(f_unwrapping),
                              CUBLAS_POINTER_MODE_HOST);
    }
    else if constexpr (Tag == TransformDispatchType::Lapack) {
      return then_with_cusolver(std::move(transfer_sender), std::move(f_unwrapping));
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
  pika::execution::experimental::start_detached(
      transform<Tag>(policy, std::forward<F>(f), std::forward<Sender>(sender)));
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
  pika::execution::experimental::start_detached(
      transformLift<Tag>(policy, std::forward<F>(f), std::forward<Ts>(ts)...));
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
  PartialTransform(PartialTransform const&) = default;
  PartialTransform& operator=(PartialTransform&&) = default;
  PartialTransform& operator=(PartialTransform const&) = default;

  template <typename Sender>
  friend auto operator|(Sender&& sender, const PartialTransform pa) {
    return transform<Tag, true, B>(pa.policy_, std::move(pa.f_), std::forward<Sender>(sender));
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
  PartialTransformDetach(PartialTransformDetach const&) = default;
  PartialTransformDetach& operator=(PartialTransformDetach&&) = default;
  PartialTransformDetach& operator=(PartialTransformDetach const&) = default;

  template <typename Sender>
  friend auto operator|(Sender&& sender, const PartialTransformDetach pa) {
    return pika::execution::experimental::start_detached(
        transform<Tag, true, B>(pa.policy_, std::move(pa.f_), std::forward<Sender>(sender)));
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
