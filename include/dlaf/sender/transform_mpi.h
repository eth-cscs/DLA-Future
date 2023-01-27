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

#include <pika/mpi.hpp>
#include <type_traits>

#include "dlaf/common/pipeline.h"
#include "dlaf/common/unwrap.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/sender/transform.h"
#include "dlaf/sender/when_all_lift.h"

namespace dlaf::comm::internal {

/// This helper "consumes" a PromiseGuard<Communicator> ensuring that after this call the one
/// passed as argument gets destroyed. All other types left as they are by the
/// second overload.
inline void consumePromiseGuardCommunicator(dlaf::common::PromiseGuard<Communicator>& pcomm) {
  [[maybe_unused]] auto pcomm_local = std::move(pcomm);
}

/// \overload consumePromiseGuardCommunicator
/// This helper "consumes" a PromiseGuard<Communicator> ensuring that after this call the one
/// passed as argument gets destroyed.
template <typename T>
void consumePromiseGuardCommunicator(T&) {}

/// Helper type for wrapping MPI calls.
///
/// Wrapper type around calls to MPI functions. Provides a call operator that
/// creates an MPI request and passes it as the last argument to the provided
/// callable. The wrapper then waits for the the request to complete with
/// yield_while.
///
/// This could in theory be a lambda inside transformMPI.  However, clang at
/// least until version 12 fails with an internal compiler error with a trailing
/// decltype for SFINAE. GCC has no problems with a lambda.
template <typename F>
struct MPIYieldWhileCallHelper {
  std::decay_t<F> f;
  template <typename... Ts>
  auto operator()(Ts&&... ts)
      -> decltype(std::move(f)(dlaf::common::internal::unwrap(ts)..., std::declval<MPI_Request*>())) {
    MPI_Request req;

    // Note:
    // Callables passed to transformMPI have their arguments passed by reference, but doing so
    // with PromiseGuard would keep the guard alive until the completion of the MPI operation,
    // whereas we are only looking to guard the submission of the MPI operation. We therefore
    // explicitly release PromiseGuard<Communicator> after submitting the MPI operation with
    // consumePromiseGuardCommunicator.
    //
    // We also use unwrap various types passed to the MPI operation, including PromiseGuards of
    // any type, to allow the MPI operation not to care whether a Communicator was wrapped in a
    // PromiseGuard or not.
    using result_type = decltype(std::move(f)(dlaf::common::internal::unwrap(ts)..., &req));
    if constexpr (std::is_void_v<result_type>) {
      std::move(f)(dlaf::common::internal::unwrap(ts)..., &req);
      (internal::consumePromiseGuardCommunicator(ts), ...);
      pika::util::yield_while(std::bind(pika::mpi::experimental::detail::eager_poll_request, req));
    }
    else {
      auto r = std::move(f)(dlaf::common::internal::unwrap(ts)..., &req);
      (internal::consumePromiseGuardCommunicator(ts), ...);
      pika::util::yield_while(std::bind(pika::mpi::experimental::detail::eager_poll_request, req));
      return r;
    }
  }
};


/// Helper type for wrapping MPI calls.
template <typename F>
struct MPICallHelper {
  std::decay_t<F> f;
  template <typename... Ts>
  auto operator()(Ts&&... ts) -> decltype(pika::unwrapping(std::move(f))(
      unwrapPromiseGuard(dlaf::internal::getReferenceWrapper(ts))...))
  {
    using result_type = decltype(pika::unwrapping(std::move(f))(
        unwrapPromiseGuard(dlaf::internal::getReferenceWrapper(ts))...));
    if constexpr (std::is_void_v<result_type>) {
      pika::unwrapping(std::move(f))(
          unwrapPromiseGuard(dlaf::internal::getReferenceWrapper(ts))...);
      (internal::consumePromiseGuardCommunicator(dlaf::internal::getReferenceWrapper(ts)), ...);
    }
    else {
      auto r = pika::unwrapping(std::move(f))(
          unwrapPromiseGuard(dlaf::internal::getReferenceWrapper(ts))...);
      (internal::consumePromiseGuardCommunicator(dlaf::internal::getReferenceWrapper(ts)), ...);
      return r;
    }
  }
};

template <typename F>
MPIYieldWhileCallHelper(F&&) -> MPIYieldWhileCallHelper<std::decay_t<F>>;

template <typename F>
MPICallHelper(F&&) -> MPICallHelper<std::decay_t<F>>;

/// Lazy transformMPI. This does not submit the work and returns a sender.
template <typename F, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
[[nodiscard]] decltype(auto) transformMPI(F&& f, Sender&& sender, pika::mpi::experimental::stream_type s) {
  namespace ex = pika::execution::experimental;
  namespace mpi = pika::mpi::experimental;

  if (mpi::get_completion_mode()==100) {
      auto snd1 = ex::transfer(std::forward<Sender>(sender), dlaf::internal::getMPIScheduler()) |
         ex::then(MPIYieldWhileCallHelper{std::forward<F>(f)});
    return ex::make_unique_any_sender(std::move(snd1));
  }
  else {
    return std::forward<Sender>(sender)
          | mpi::transform_mpi(MPICallHelper{std::forward<F>(f)}, s);
  }
}

/// Fire-and-forget transformMPI. This submits the work and returns void.
template <typename F, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
void transformMPIDetach(F&& f, Sender&& sender, pika::mpi::experimental::stream_type s) {
  pika::execution::experimental::start_detached(
      transformMPI(std::forward<F>(f), std::forward<Sender>(sender), s));
}

/// Lazy transformMPI. This does not submit the work and returns a sender. First
/// lifts non-senders into senders using just, and then calls transform with a
/// when_all sender of the lifted senders.
template <typename F, typename... Ts>
[[nodiscard]] decltype(auto) transformMPILift(F&& f, pika::mpi::experimental::stream_type s, Ts&&... ts) {
  return transformMPI(std::forward<F>(f), dlaf::internal::whenAllLift(std::forward<Ts>(ts)...), s);
}

/// Fire-and-forget transformMPI. This submits the work and returns void. First
/// lifts non-senders into senders using just, and then calls transform with a
/// when_all sender of the lifted senders.
template <typename F, typename... Ts>
void transformMPILiftDetach(F&& f, pika::mpi::experimental::stream_type s, Ts&&... ts) {
  pika::execution::experimental::start_detached(
      transformLift(std::forward<F>(f), s, std::forward<Ts>(ts)...));
}

template <typename F>
struct PartialTransformMPIBase {
  std::decay_t<F> f_;
  pika::mpi::experimental::stream_type s_;
};

/// A partially applied transformMPI, with the callable object given, but the
/// predecessor sender missing. The predecessor sender is applied when calling
/// the operator| overload.
template <typename F>
class PartialTransformMPI : private PartialTransformMPIBase<F> {
public:
  template <typename F_>
  PartialTransformMPI(F_&& f, pika::mpi::experimental::stream_type s) :
      PartialTransformMPIBase<F>{std::forward<F_>(f), s} {}
  PartialTransformMPI(PartialTransformMPI&&) = default;
  PartialTransformMPI(PartialTransformMPI const&) = default;
  PartialTransformMPI& operator=(PartialTransformMPI&&) = default;
  PartialTransformMPI& operator=(PartialTransformMPI const&) = default;

  template <typename Sender>
  friend auto operator|(Sender&& sender, const PartialTransformMPI pa) {
    return transformMPI(std::move(pa.f_), std::forward<Sender>(sender), pa.s_);
  }
};

template <typename F>
PartialTransformMPI(F&& f, pika::mpi::experimental::stream_type) -> PartialTransformMPI<std::decay_t<F>>;

/// A partially applied transformMPIDetach, with the callable object given, but
/// the predecessor sender missing. The predecessor sender is applied when
/// calling the operator| overload.
template <typename F>
class PartialTransformMPIDetach : private PartialTransformMPIBase<F> {
public:
  template <typename F_>
  PartialTransformMPIDetach(F_&& f) : PartialTransformMPIBase<F>{std::forward<F_>(f)} {}
  PartialTransformMPIDetach(PartialTransformMPIDetach&&) = default;
  PartialTransformMPIDetach(PartialTransformMPIDetach const&) = default;
  PartialTransformMPIDetach& operator=(PartialTransformMPIDetach&&) = default;
  PartialTransformMPIDetach& operator=(PartialTransformMPIDetach const&) = default;

  template <typename Sender>
  friend auto operator|(Sender&& sender, const PartialTransformMPIDetach pa) {
    return pika::execution::experimental::start_detached(
        transformMPI(std::move(pa.f_), std::forward<Sender>(sender), pa.s_));
  }
};

template <typename F>
PartialTransformMPIDetach(F&& f) -> PartialTransformMPIDetach<std::decay_t<F>>;

/// \overload transformMPI
///
/// This overload partially applies the MPI transform for later use with
/// operator| with a sender on the left-hand side.
template <typename F>
[[nodiscard]] decltype(auto) transformMPI(F&& f, pika::mpi::experimental::stream_type s) {
  return PartialTransformMPI{std::forward<F>(f), s};
}

/// \overload transformMPIDetach
///
/// This overload partially applies transformMPIDetach for later use with
/// operator| with a sender on the left-hand side.
template <typename F>
[[nodiscard]] decltype(auto) transformMPIDetach(F&& f, pika::mpi::experimental::stream_type s) {
  return PartialTransformMPIDetach{std::forward<F>(f), s};
}
}
