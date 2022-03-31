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

#include <pika/mpi.hpp>
#include <pika/unwrap.hpp>

#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/sender/transform.h"
#include "dlaf/sender/when_all_lift.h"

namespace dlaf::comm::internal {

/// Helper for moving a PromiseGuard<Communicator> into the MPI function being
/// called by transformMPI. Callables passed to transformMPI have their
/// arguments passed by reference, but doing so with PromiseGuard would keep the
/// guard alive until the completion of the MPI operation, whereas we are only
/// looking to guard the submission of the MPI operation. We therefore use this
/// wrapper to move PromiseGuard<Communicator> into a transformMPI callables so
/// that the guard is released on return from the callable.
template <typename T>
decltype(auto) movePromiseGuard(T&& t) {
  if constexpr (std::is_same_v<std::decay_t<T>, dlaf::common::PromiseGuard<Communicator>>) {
    return std::move(t);
  }
  else {
    return static_cast<T&>(t);
  }
}

/// Lazy transformMPI. This does not submit the work and returns a sender.
template <typename F, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
[[nodiscard]] decltype(auto) transformMPI(F&& f, Sender&& sender) {
  using pika::execution::experimental::then;
  using pika::execution::experimental::thread_pool_scheduler;
  using pika::execution::experimental::transfer;
  using pika::unwrapping;

  auto f_wrapper = [f = std::forward<F>(f)](auto&&... ts)
      -> decltype(unwrapping(std::move(f))(movePromiseGuard(ts)..., std::declval<MPI_Request*>())) {
    MPI_Request req;
    auto is_request_completed = [&req] {
      int flag;
      MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
      return flag == 0;
    };

    using result_type = decltype(unwrapping(std::move(f))(movePromiseGuard(ts)..., &req));
    if constexpr (std::is_void_v<result_type>) {
      unwrapping(std::move(f))(movePromiseGuard(ts)..., &req);
      pika::util::yield_while(is_request_completed);
    }
    else {
      auto r = unwrapping(std::move(f))(movePromiseGuard(ts)..., &req);
      pika::util::yield_while(is_request_completed);
      return r;
    }
  };
  return transfer(std::forward<Sender>(sender), dlaf::internal::getMPIScheduler()) |
         then(std::move(f_wrapper));
}

/// Fire-and-forget transformMPI. This submits the work and returns void.
template <typename F, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
void transformMPIDetach(F&& f, Sender&& sender) {
  pika::execution::experimental::start_detached(
      transformMPI(std::forward<F>(f), std::forward<Sender>(sender)));
}

/// Lazy transformMPI. This does not submit the work and returns a sender. First
/// lifts non-senders into senders using just, and then calls transform with a
/// when_all sender of the lifted senders.
template <typename F, typename... Ts>
[[nodiscard]] decltype(auto) transformMPILift(F&& f, Ts&&... ts) {
  return transformMPI(std::forward<F>(f), dlaf::internal::whenAllLift(std::forward<Ts>(ts)...));
}

/// Fire-and-forget transformMPI. This submits the work and returns void. First
/// lifts non-senders into senders using just, and then calls transform with a
/// when_all sender of the lifted senders.
template <typename F, typename... Ts>
void transformMPILiftDetach(F&& f, Ts&&... ts) {
  pika::execution::experimental::start_detached(
      transformLift(std::forward<F>(f), std::forward<Ts>(ts)...));
}

template <typename F>
struct PartialTransformMPIBase {
  std::decay_t<F> f_;
};

/// A partially applied transformMPI, with the callable object given, but the
/// predecessor sender missing. The predecessor sender is applied when calling
/// the operator| overload.
template <typename F>
class PartialTransformMPI : private PartialTransformMPIBase<F> {
public:
  template <typename F_>
  PartialTransformMPI(F_&& f) : PartialTransformMPIBase<F>{std::forward<F_>(f)} {}
  PartialTransformMPI(PartialTransformMPI&&) = default;
  PartialTransformMPI(PartialTransformMPI const&) = default;
  PartialTransformMPI& operator=(PartialTransformMPI&&) = default;
  PartialTransformMPI& operator=(PartialTransformMPI const&) = default;

  template <typename Sender>
  friend auto operator|(Sender&& sender, const PartialTransformMPI pa) {
    return transformMPI(std::move(pa.f_), std::forward<Sender>(sender));
  }
};

template <typename F>
PartialTransformMPI(F&& f) -> PartialTransformMPI<std::decay_t<F>>;

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
        transformMPI(std::move(pa.f_), std::forward<Sender>(sender)));
  }
};

template <typename F>
PartialTransformMPIDetach(F&& f) -> PartialTransformMPIDetach<std::decay_t<F>>;

/// \overload transformMPI
///
/// This overload partially applies the MPI transform for later use with
/// operator| with a sender on the left-hand side.
template <typename F>
[[nodiscard]] decltype(auto) transformMPI(F&& f) {
  return PartialTransformMPI{std::forward<F>(f)};
}

/// \overload transformMPIDetach
///
/// This overload partially applies transformMPIDetach for later use with
/// operator| with a sender on the left-hand side.
template <typename F>
[[nodiscard]] decltype(auto) transformMPIDetach(F&& f) {
  return PartialTransformMPIDetach{std::forward<F>(f)};
}
}
