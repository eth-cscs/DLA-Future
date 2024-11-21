//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <type_traits>
#include <utility>

#include <pika/execution.hpp>
#include <pika/mpi.hpp>

#include <dlaf/common/consume_rvalues.h>
#include <dlaf/common/unwrap.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/sender/continues_on.h>
#include <dlaf/sender/transform.h>

namespace dlaf::comm::internal {

/// This helper "consumes" a CommunicatorPipelineExclusiveWrapper ensuring that after this call
/// the one passed as argument gets destroyed. All other types left as they are
/// by the second overload.
inline void consumeCommunicatorWrapper(CommunicatorPipelineExclusiveWrapper& comm_wrapper) {
  [[maybe_unused]] auto comm_wrapper_local = std::move(comm_wrapper);
}

/// \overload consumeCommunicatorWrapper (for non communicator types)
template <typename T>
void consumeCommunicatorWrapper(T&) {}

/// Helper type for wrapping MPI calls.
///
/// The wrapper explicitly releases any dla communicator objects when the pika::transform_mpi
/// function returns (e.g. a message has been sent/posted) to prevent blocking access to many
/// queued mpi operations.
/// The mpi operations can complete asynchronously later, but the commmunicator is
/// released/made available once the mpi task has been safely initiated
///
/// This could in theory be a lambda inside transformMPI.  However, clang at
/// least until version 12 fails with an internal compiler error with a trailing
/// decltype for SFINAE. GCC has no problems with a lambda.
template <typename F>
struct MPICallHelper {
  std::decay_t<F> f;

  template <typename... Ts>
  auto operator()(Ts&&... ts) -> decltype(std::move(f)(dlaf::common::internal::unwrap(ts)...)) {
    using result_type = decltype(std::move(f)(dlaf::common::internal::unwrap(ts)...));
    if constexpr (std::is_void_v<result_type>) {
      std::move(f)(dlaf::common::internal::unwrap(ts)...);
      (internal::consumeCommunicatorWrapper(ts), ...);
    }
    else {
      auto r = std::move(f)(dlaf::common::internal::unwrap(ts)...);
      (internal::consumeCommunicatorWrapper(ts), ...);
      return r;
    }
  }
};

template <typename F>
MPICallHelper(F&&) -> MPICallHelper<std::decay_t<F>>;

/// Lazy transformMPI. Returns a sender that will submit the work passed in
template <typename F, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
[[nodiscard]] decltype(auto) transformMPI(F&& f, Sender&& sender) {
  using dlaf::common::internal::ConsumeRvalues;
  using pika::execution::experimental::drop_operation_state;
  using pika::mpi::experimental::transform_mpi;
  return std::forward<Sender>(sender)                                        //
         | transform_mpi(ConsumeRvalues{MPICallHelper{std::forward<F>(f)}})  //
         | drop_operation_state();
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
  PartialTransformMPI(const PartialTransformMPI&) = default;
  PartialTransformMPI& operator=(PartialTransformMPI&&) = default;
  PartialTransformMPI& operator=(const PartialTransformMPI&) = default;

  template <typename Sender>
  friend auto operator|(Sender&& sender, PartialTransformMPI pa) {
    return transformMPI(std::move(pa.f_), std::forward<Sender>(sender));
  }
};

template <typename F>
PartialTransformMPI(F&& f) -> PartialTransformMPI<std::decay_t<F>>;

/// \overload transformMPI
///
/// This overload partially applies the MPI transform for later use with
/// operator| with a sender on the left-hand side.
template <typename F>
[[nodiscard]] decltype(auto) transformMPI(F&& f) {
  return PartialTransformMPI{std::forward<F>(f)};
}
}
