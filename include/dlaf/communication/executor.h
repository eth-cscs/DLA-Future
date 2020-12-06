//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

/// @file

#include <atomic>
#include <memory>
#include <utility>

#include <mpi.h>

#include <hpx/async_mpi/mpi_executor.hpp>
#include <hpx/async_mpi/mpi_future.hpp>
#include <hpx/config.hpp>
#include <hpx/future.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/thread_executors.hpp>
#include <hpx/modules/execution_base.hpp>

#include "dlaf/communication/communicator.h"
#include "dlaf/communication/init.h"

namespace dlaf {
namespace comm {

/// An executor based on John's upstream HPX polling mechanism.
class mpi_polling_executor {
  Communicator comm_;

public:
  // Associate the parallel_execution_tag executor tag type as a default with this executor.
  using execution_category = hpx::parallel::execution::parallel_execution_tag;

  mpi_polling_executor(Communicator comm) : comm_(std::move(comm)) {}
  mpi_polling_executor(std::string, Communicator comm) : comm_(std::move(comm)) {}

  constexpr bool operator==(const mpi_polling_executor& rhs) const noexcept {
    return comm_ == rhs.comm_;
  }

  constexpr bool operator!=(const mpi_polling_executor& rhs) const noexcept {
    return !(*this == rhs);
  }

  constexpr const mpi_polling_executor& context() const noexcept {
    return *this;
  }

  Communicator comm() const noexcept {
    return comm_;
  }

  template <typename F, typename... Ts>
  hpx::future<void> async_execute(F f, Ts... ts) noexcept {
    return hpx::mpi::experimental::detail::async(f, ts..., comm_);
  }
};

/// An executor for MPI calls.
class executor {
  Communicator comm_;
  hpx::threads::executors::pool_executor ex_;

public:
  // Associate the parallel_execution_tag executor tag type as a default with this executor.
  using execution_category = hpx::parallel::execution::parallel_execution_tag;

  executor(Communicator comm)
      : comm_(std::move(comm)), ex_("default", hpx::threads::thread_priority_high) {}

  executor(std::string pool, Communicator comm)
      : comm_(std::move(comm)), ex_(pool, hpx::threads::thread_priority_high) {}

  bool operator==(const executor& rhs) const noexcept {
    return comm_ == rhs.comm_ && ex_ == rhs.ex_;
  }

  bool operator!=(const executor& rhs) const noexcept {
    return !(*this == rhs);
  }

  const executor& context() const noexcept {
    return *this;
  }

  Communicator comm() const noexcept {
    return comm_;
  }

  /// The function only accepts non-blocking MPI routines. Both MPI_THREAD_MULTIPLE and
  /// MPI_THREAD_SERIALIZED are supported.
  ///
  /// Example usage:
  ///
  /// ```
  /// ex.async_execute(MPI_Irecv, recv_ptr, num_recv_elems, MPI_DOUBLE, src_rank, tag);
  /// ```
  ///
  /// compared to the usual MPI non-blocking call:
  ///
  /// ```
  /// MPI_Irecv(recv_ptr, num_recv_elems, MPI_DOUBLE, src_rank, tag, comm, &request);
  /// ```
  ///
  /// there are two differences
  ///
  /// 1) MPI_Comm (comm) doesn't need to be specified as it is passed by the executor
  /// 2) MPI_Request (request) is omitted as it is handled internally in async()
  ///
  template <typename F, typename... Ts>
  hpx::future<void> async_execute(F f, Ts... ts) noexcept {
    // Note that the call is made inline instead of being wrapped in a task. This is to avoid overheads
    // with task creations and to allow MPI calls on the same executor/communicator to be easily ordered
    // within the task they are made.
    MPI_Request req;
    mpi_invoke(f, ts..., comm_, &req);

    return hpx::async(ex_, [req]() mutable {
      // Yield until non-blocking communication completes.
      hpx::util::yield_while([&req] {
        int flag;
        mpi_invoke(MPI_Test, &req, &flag, MPI_STATUS_IGNORE);
        return flag == 0;
      });
    });
  }
};

}
}

namespace hpx {
namespace parallel {
namespace execution {

template <>
struct is_two_way_executor<dlaf::comm::executor> : std::true_type {};

}
}
}
