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
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/schedulers/shared_priority_queue_scheduler.hpp>
#include <hpx/synchronization/mutex.hpp>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include <hpx/async_mpi/mpi_future.hpp>
#include <hpx/execution.hpp>
#include <hpx/functional.hpp>
#include <hpx/future.hpp>
#include <hpx/include/parallel_executors.hpp>

#include "dlaf/communication/communicator.h"
#include "dlaf/communication/init.h"

namespace dlaf {
namespace comm {

struct pool_hints_manager {
  std::string name;
  int nthreads;
  std::unique_ptr<bool[]> hints_arr;
  hpx::lcos::local::mutex mtx;

  // returns the local thread number of the first free thread, if no such thread is found, returns -1
  int get_free_thread() {
    std::lock_guard<hpx::lcos::local::mutex> lk(mtx);
    for (int i = 0; i < nthreads; ++i) {
      if (hints_arr[i]) {
        hints_arr[i] = false;
        return i;
      }
    }
    return -1;
  }
};

//pool_hints_manager* get_pool_hints(const std::string& pool_name) {
//  static std::vector<pool_hints_manager> managers_arr;  // TODO: init
//  for (auto& manager : managers_arr) {
//
//  }
//  // TODO: find pool_name
//}

enum class MPIMech { Blocking, Polling, Yielding };

namespace detail {

template <MPIMech mech>
struct nbmpi_internal_mech {};

template <>
struct nbmpi_internal_mech<MPIMech::Polling> {
  void operator()(MPI_Request req) {
    hpx::mpi::experimental::get_future(req).get();
  }
};

template <>
struct nbmpi_internal_mech<MPIMech::Yielding> {
  void operator()(MPI_Request req) {
    hpx::util::yield_while([&req] {
      int flag;
      mpi_invoke(MPI_Test, &req, &flag, MPI_STATUS_IGNORE);
      return flag == 0;
    });
  }
};

// Non-blocking
template <MPIMech M, class F, class... Ts>
struct executor_launch_impl {
  void operator()(hpx::future<void>, hpx::lcos::local::promise<void> p, Communicator comm, F f,
                  Ts... ts) noexcept {
    MPI_Request req;
    hpx::invoke(std::move(f), std::move(ts)..., comm, &req);
    p.set_value();
    nbmpi_internal_mech<M>{}(req);
  }
};

// Blocking
template <class F, class... Ts>
struct executor_launch_impl<MPIMech::Blocking, F, Ts...> {
  void operator()(hpx::future<void>, hpx::lcos::local::promise<void> p, Communicator comm, F f,
                  Ts... ts) noexcept {
    hpx::invoke(std::move(f), std::move(ts)..., comm);
    p.set_value();
  }
};

template <MPIMech M>
struct hint_manager {};

// TODO: change hints
template <>
struct hint_manager<MPIMech::Blocking> {
  // get a hint
  hpx::threads::thread_schedule_hint get_hint() const {
    return hpx::threads::thread_schedule_hint{0};
  }
};

template <MPIMech M>
hpx::execution::parallel_executor init_exec(const std::string& pool, hint_manager<M>) {
  return hpx::execution::parallel_executor(&hpx::resource::get_thread_pool(pool));
}

inline hpx::execution::parallel_executor init_exec(const std::string& pool,
                                                   const hint_manager<MPIMech::Blocking>& mgr) {
  return hpx::execution::parallel_executor(&hpx::resource::get_thread_pool(pool),
                                           hpx::threads::thread_priority::default_,
                                           hpx::threads::thread_stacksize::nostack, mgr.get_hint());
}

}

template <MPIMech M>
class Executor {
  Communicator comm_;
  hpx::future<void> tail_;
  detail::hint_manager<M> mgr_;
  hpx::execution::parallel_executor ex_;

public:
  // Notes:
  //   1. `comm` should not be used by other executors
  //   2. MPI event polling has to be enabled for `MPIMech::Polling`.
  Executor(const std::string& pool, Communicator comm)
      : comm_(std::move(comm)), tail_(hpx::make_ready_future<void>()), mgr_{},
        ex_(detail::init_exec(pool, mgr_)) {
    ;
  }

  Executor(const Executor& o) = delete;
  Executor& operator=(const Executor& o) = delete;

  Executor(Executor&&) = default;
  Executor& operator=(Executor&&) = default;

  ~Executor() {
    if (tail_.valid())
      tail_.get();
  }

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
  decltype(auto) async_execute(F f, Ts... ts) noexcept {
    hpx::lcos::local::promise<void> promise_next;
    auto before_last = std::move(tail_);
    tail_ = promise_next.get_future();
    return hpx::dataflow(ex_, detail::executor_launch_impl<M, F, Ts...>{}, std::move(before_last),
                         std::move(promise_next), comm_, std::move(f), std::move(ts)...);
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
