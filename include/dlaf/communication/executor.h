//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
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
#include <hpx/modules/execution_base.hpp>

#include "dlaf/communication/communicator.h"
#include "dlaf/communication/init.h"

namespace dlaf {
namespace comm {

enum class MPIMech { Blocking, Polling, Yielding };

namespace detail {

template <MPIMech mech>
struct executor_launch_impl {};

template <>
struct executor_launch_impl<MPIMech::Polling> {
  // TODO: check if polling was enabled
  executor_launch_impl(const std::string&) {}
  hpx::future<void> get_future(MPI_Request req) noexcept {
    return hpx::mpi::experimental::get_future(req);
  }
};

template <>
struct executor_launch_impl<MPIMech::Yielding> {
  hpx::execution::parallel_executor ex_;
  executor_launch_impl(const std::string& pool)
      : ex_(&hpx::resource::get_thread_pool(pool), hpx::threads::thread_priority::high) {}
  hpx::future<void> get_future(MPI_Request req) noexcept {
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

template <MPIMech M>
class Executor {
  detail::executor_launch_impl<M> launcher_;
  Communicator comm_;

public:
  Executor(const std::string& pool, Communicator comm) : launcher_(pool), comm_(std::move(comm)) {}

  bool operator==(const Executor& rhs) const noexcept {
    return comm_ == rhs.comm_;
  }

  bool operator!=(const Executor& rhs) const noexcept {
    return !(*this == rhs);
  }

  const Executor& context() const noexcept {
    return *this;
  }

  Communicator comm() const noexcept {
    return comm_;
  }

  template <typename F, typename... Ts>
  hpx::future<void> async_execute(F&& f, Ts&&... ts) noexcept {
    MPI_Request req;
    mpi_invoke(std::forward<F>(f), std::forward<Ts>(ts)..., comm_, &req);
    return launcher_.get_future(req);
  }
};
}
}

namespace hpx {
namespace parallel {
namespace execution {

template <dlaf::comm::MPIMech M>
struct is_two_way_executor<dlaf::comm::Executor<M>> : std::true_type {};

}
}
}
