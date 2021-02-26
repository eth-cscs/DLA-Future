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
#include <hpx/runtime_local/thread_pool_helpers.hpp>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

#include <hpx/async_mpi/mpi_future.hpp>
#include <hpx/execution.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/functional.hpp>
#include <hpx/future.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/synchronization/mutex.hpp>

#include <mpi.h>

#include "dlaf/common/assert.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/init.h"

namespace dlaf {
namespace comm {

enum class MPIMech { Blocking, Polling, Yielding };

namespace detail {

inline std::atomic<bool>* get_hints_mask() {
  using hints_arr_t = std::unique_ptr<std::atomic<bool>[]>;
  static hints_arr_t hints_mask = []() {
    std::size_t nthreads = hpx::resource::get_num_threads();
    hints_arr_t hints(new std::atomic<bool>[nthreads]);
    for (int i = 0; i < nthreads; ++i) {
      hints[i].store(true);
    }
    return hints;
  }();
  return hints_mask.get();
}

inline int get_free_thread_index(const std::string& pool_name) {
  int thread_offset = 0;
  for (int i_pool = 0; i_pool < hpx::resource::get_pool_index(pool_name); ++i_pool) {
    thread_offset += hpx::resource::get_num_threads(i_pool);
  };

  std::atomic<bool>* hints_mask = get_hints_mask();
  for (int i_thd = 0; i_thd < hpx::resource::get_num_threads(pool_name); ++i_thd) {
    int index = i_thd + thread_offset;
    if (hints_mask[index].load()) {
      hints_mask[index].store(false);
      return index;
    }
  }
  return -1;
}

inline bool is_stealing_enabled(const std::string& pool_name) {
  return hpx::resource::get_thread_pool(pool_name).get_scheduler()->has_scheduler_mode(
      hpx::threads::policies::scheduler_mode(
          hpx::threads::policies::scheduler_mode::enable_stealing |
          hpx::threads::policies::scheduler_mode::enable_stealing_numa));
}

template <MPIMech M>
struct hint_manager {
  hint_manager(const std::string&) {}
};

template <>
struct hint_manager<MPIMech::Blocking> {
  int index_;

public:
  hint_manager(const std::string& pool_name) {
    using hpx::resource::get_num_threads;
    // Assert that the pool has task stealing disabled
    DLAF_ASSERT(!is_stealing_enabled(pool_name) || get_num_threads(pool_name) == 1, pool_name);
    hpx::util::yield_while([this, &pool_name] {
      index_ = get_free_thread_index(pool_name);
      return index_ == -1;
    });
  }
  hint_manager& operator=(hint_manager&& o) {
    index_ = o.index_;
    o.index_ = -1;
    return *this;
  }
  hint_manager(hint_manager&& o) {
    *this = std::move(o);
  }
  ~hint_manager() {
    if (index_ != -1) {
      get_hints_mask()[index_].store(true);
    }
  }

  int get_thread_index() const {
    return index_;
  }
};

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
hpx::execution::parallel_executor init_exec(const std::string& pool, const hint_manager<M>&) {
  return hpx::execution::parallel_executor(&hpx::resource::get_thread_pool(pool));
}

inline hpx::execution::parallel_executor init_exec(const std::string& pool,
                                                   const hint_manager<MPIMech::Blocking>& mgr) {
  return hpx::execution::parallel_executor(&hpx::resource::get_thread_pool(pool),
                                           hpx::threads::thread_priority::default_,
                                           hpx::threads::thread_stacksize::nostack,
                                           hpx::threads::thread_schedule_hint(mgr.get_thread_index()));
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
      : comm_(std::move(comm)), tail_(hpx::make_ready_future<void>()), mgr_(pool),
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
    // Note:: `ts...` and `f` are value parameters to avoid compilation issues with forwarding universal
    // references (i.e. `std::forward<>()` a &&) in `hpx::dataflow()`.
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
