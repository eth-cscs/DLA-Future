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

#ifdef EXTRA_MPI_TYPES_DEBUGGING
#include <pika/debugging/demangle_helper.hpp>
#endif
#include <pika/debugging/print.hpp>
#include <pika/execution.hpp>

#include <dlaf/common/consume_rvalues.h>
#include <dlaf/common/unwrap.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/sender/transform.h>
#include <dlaf/sender/when_all_lift.h>
//
#include <pika/mpi.hpp>

namespace dlaf::comm::internal {

template <int Level>
static pika::debug::detail::print_threshold<Level, 0> dla_debug("DLA_MPI");

/// This helper "consumes" a CommunicatorPipelineExclusiveWrapper ensuring that after this call
/// the one passed as argument gets destroyed. All other types left as they are
/// by the second overload.
inline void consumeCommunicatorWrapper(CommunicatorPipelineExclusiveWrapper& comm_wrapper) {
  [[maybe_unused]] auto comm_wrapper_local = std::move(comm_wrapper);
}

/// \overload consumeCommunicatorWrapper
template <typename T>
void consumeCommunicatorWrapper(T&) {}

/// Helper type for wrapping MPI calls.
///
/// Wrapper type around calls to MPI functions. Provides a call operator that
/// creates an MPI request and passes it as the last argument to the provided
/// callable. The wrapper then waits for the request to complete with
/// yield_while.
///
/// This could in theory be a lambda inside transformMPI.  However, clang at
/// least until version 12 fails with an internal compiler error with a trailing
/// decltype for SFINAE. GCC has no problems with a lambda.
template <typename F>
struct MPIYieldWhileCallHelper {
  std::decay_t<F> f;
  template <typename... Ts>
  auto operator()(Ts&&... ts) {
    namespace mpid = pika::mpi::experimental::detail;
    MPI_Request req;

    // Note:
    // Callables passed to transformMPI have their arguments passed by reference, but doing so
    // with PromiseGuard would keep the guard alive until the completion of the MPI operation,
    // whereas we are only looking to guard the submission of the MPI operation. We therefore
    // explicitly release CommunicatorPipelineExclusiveWrapper after submitting the MPI operation
    // with consumeCommunicatorWrapper.
    //
    // We also use unwrap various types passed to the MPI operation, including PromiseGuards of
    // any type, to allow the MPI operation not to care whether a Communicator was wrapped in a
    // PromiseGuard or not.
    using result_type = decltype(std::move(f)(dlaf::common::internal::unwrap(ts)..., &req));
    if constexpr (std::is_void_v<result_type>) {
      std::move(f)(dlaf::common::internal::unwrap(ts)..., &req);
      (internal::consumeCommunicatorWrapper(ts), ...);
      pika::util::yield_while([req]() { return !mpid::poll_request(req); });
    }
    else {
      /*auto r = */ std::move(f)(dlaf::common::internal::unwrap(ts)..., &req);
      (internal::consumeCommunicatorWrapper(ts), ...);
      pika::util::yield_while([req]() { return !mpid::poll_request(req); });
    }
  }
};

/// Helper type for wrapping MPI calls.
template <typename F>
struct MPICallHelper {
  std::decay_t<F> f;

  template <typename... Ts>
  auto operator()(Ts&&... ts) -> decltype(std::move(f)(dlaf::common::internal::unwrap(ts)...)) {
    using namespace pika::debug::detail;
    PIKA_DETAIL_DP(dla_debug<5>, debug(str<>("MPICallHelper"), pika::debug::print_type<Ts...>(", ")));
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
MPIYieldWhileCallHelper(F&&) -> MPIYieldWhileCallHelper<std::decay_t<F>>;

template <typename F>
MPICallHelper(F&&) -> MPICallHelper<std::decay_t<F>>;

/// Lazy transformMPI. This does not submit the work and returns a sender.
template <typename F, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
[[nodiscard]] decltype(auto) transformMPI(F&& f, Sender&& sender) {
  namespace ex = pika::execution::experimental;
  namespace mpi = pika::mpi::experimental;
  namespace mpid = pika::mpi::experimental::detail;

  if (mpi::get_completion_mode() >= static_cast<int>(mpid::handler_mode::unspecified)) {
    auto snd1 =
        ex::transfer(std::forward<Sender>(sender), dlaf::internal::getMPIScheduler()) |
        ex::then(dlaf::common::internal::ConsumeRvalues{MPIYieldWhileCallHelper{std::forward<F>(f)}});
    return ex::make_unique_any_sender(std::move(snd1));
  }
  else {
#ifdef EXTRA_MPI_TYPES_DEBUGGING
    auto snd1 =
        std::forward<Sender>(sender) |
        ex::let_value([=, f = std::move(f)]<typename... LArgs>(LArgs&&... largs) {
          PIKA_DETAIL_DP(dla_debug<2>, debug(str<>("Args to MPI fn\n"),
                                             pika::debug::print_type<LArgs...>(", "), "\nValues\n"));
          return ex::just(std::move(largs)...) |
                 mpi::transform_mpi(dlaf::common::internal::ConsumeRvalues{MPICallHelper{std::move(f)}});
        });
    return ex::make_unique_any_sender(std::move(snd1));
#else
    PIKA_DETAIL_DP(dla_debug<5>, debug(str<>("MPI fn\n")));
    auto snd1 =
        std::forward<Sender>(sender) |
        mpi::transform_mpi(dlaf::common::internal::ConsumeRvalues{MPICallHelper{std::forward<F>(f)}});
    return ex::make_unique_any_sender(std::move(snd1));
#endif
  }
}

/// Fire-and-forget transformMPI. This submits the work and returns void.
template <typename F, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
void transformMPIDetach(F&& f, Sender&& sender) {
  pika::execution::experimental::start_detached(transformMPI(std::forward<F>(f),
                                                             std::forward<Sender>(sender)));
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
  pika::execution::experimental::start_detached(transformLift(std::forward<F>(f),
                                                              std::forward<Ts>(ts)...));
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
