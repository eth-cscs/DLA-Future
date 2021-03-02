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
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/runtime_local/thread_pool_helpers.hpp>
#include <hpx/type_support/unused.hpp>
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
#include "dlaf/communication/mech.h"

namespace dlaf {
namespace comm {

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
  hint_manager(const hint_manager& o) = default;
  hint_manager& operator=(const hint_manager& o) = default;
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

template <>
struct nbmpi_internal_mech<MPIMech::Blocking> {
  void operator()(MPI_Request) {}
};

template <MPIMech M>
struct extra_params {
  hpx::tuple<Communicator, MPI_Request*> operator()(Communicator comm, MPI_Request* req_ptr) {
    return hpx::make_tuple(std::move(comm), req_ptr);
  }
};

template <>
struct extra_params<MPIMech::Blocking> {
  hpx::tuple<Communicator> operator()(Communicator comm, MPI_Request*) {
    return hpx::make_tuple(std::move(comm));
  }
};

template <class R>
struct invoke_fused_wrapper {
  R val;
  template <class F, class TupleArgs>
  invoke_fused_wrapper(F&& f, TupleArgs&& ts) {
    val = hpx::invoke_fused(std::forward<F>(f), std::forward<TupleArgs>(ts));
  }
  R async_return() {
    return std::move(val);
  }
  R dataflow_return() {
    return std::move(val);
  }
};

template <>
struct invoke_fused_wrapper<void> {
  template <class F, class TupleArgs>
  invoke_fused_wrapper(F&& f, TupleArgs&& ts) {
    hpx::invoke_fused(std::forward<F>(f), std::forward<TupleArgs>(ts));
  }
  void async_return() {}
  auto dataflow_return() {
    return hpx::util::unused;
  }
};

// Non-blocking
template <MPIMech M, class F, class... Ts>
struct async_helper_fn {
  using result_t = typename hpx::util::invoke_result<F, Ts..., Communicator, MPI_Request*>::type;
  void operator()(std::true_type, hpx::future<void>, hpx::lcos::local::promise<void> p,
                  Communicator comm, F&& f, Ts&&... ts) noexcept {
    MPI_Request req;
    hpx::invoke(std::forward<F>(f), std::forward<Ts>(ts)..., std::move(comm), &req);
    p.set_value();
    nbmpi_internal_mech<M>{}(req);
  }
  auto operator()(std::false_type, hpx::future<void>, hpx::lcos::local::promise<void> p,
                  Communicator comm, F&& f, Ts&&... ts) noexcept {
    MPI_Request req;
    auto r = hpx::invoke(std::forward<F>(f), std::forward<Ts>(ts)..., std::move(comm), &req);
    p.set_value();
    nbmpi_internal_mech<M>{}(req);
    return r;
  }
};

// Blocking
template <class F, class... Ts>
struct async_helper_fn<MPIMech::Blocking, F, Ts...> {
  using result_t = typename hpx::util::invoke_result<F, Ts..., Communicator>::type;
  void operator()(std::true_type, hpx::future<void>, hpx::lcos::local::promise<void> p,
                  Communicator comm, F&& f, Ts&&... ts) noexcept {
    hpx::invoke(std::forward<F>(f), std::forward<Ts>(ts)..., comm);
    p.set_value();
  }
  decltype(auto) operator()(std::false_type, hpx::future<void>, hpx::lcos::local::promise<void> p,
                            Communicator comm, F&& f, Ts&&... ts) noexcept {
    auto r = hpx::invoke(std::forward<F>(f), std::forward<Ts>(ts)..., comm);
    p.set_value();
    return r;
  }
};

}

template <MPIMech M>
class Executor {
  struct task_chain;  // forward declaration

  Communicator comm_;
  std::shared_ptr<task_chain> tc_ptr;
  detail::hint_manager<M> mgr_;
  hpx::execution::parallel_executor ex_;

  struct task_chain {
    hpx::future<void> tail;
    mutable hpx::lcos::local::mutex mt;

    task_chain() : tail(hpx::make_ready_future<void>()) {}
  };

  struct task_chain_deleter {
    void operator()(task_chain* tc) {
      if (tc->tail.valid()) {
        tc->tail.get();
      }
    }
  };

public:
  // Notes:
  //   1. `comm` should not be used by other executors
  //   2. MPI event polling has to be enabled for `MPIMech::Polling`.
  Executor(const std::string& pool, Communicator comm)
      : comm_(std::move(comm)),
        tc_ptr(std::shared_ptr<task_chain>(new task_chain(), task_chain_deleter())), mgr_(pool),
        ex_(detail::init_exec(pool, mgr_)) {
    ;
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
  auto async_execute(F&& f, Ts&&... ts) noexcept {
    using is_void = typename std::is_void<typename detail::async_helper_fn<M, F, Ts...>::result_t>::type;
    hpx::future<void> before_last;
    hpx::lcos::local::promise<void> promise_next;
    {
      std::lock_guard<hpx::lcos::local::mutex> lk(tc_ptr->mt);
      before_last = std::move(tc_ptr->tail);
      tc_ptr->tail = promise_next.get_future();
    }
    return hpx::dataflow(ex_,
                         detail::async_helper_fn<M, typename std::decay<F>::type,
                                                 typename std::decay<Ts>::type...>(),
                         is_void{}, std::move(before_last), std::move(promise_next), comm_,
                         std::forward<F>(f), std::forward<Ts>(ts)...);
  }

  template <class Frame, class F, class TupleArgs>
  void dataflow_finalize(Frame&& frame, F&& f, TupleArgs&& args) {
    hpx::future<void> before_last;
    hpx::lcos::local::promise<void> promise_next;
    {
      std::lock_guard<hpx::lcos::local::mutex> lk(tc_ptr->mt);
      before_last = std::move(tc_ptr->tail);
      tc_ptr->tail = promise_next.get_future();
    }

    // Ensure the dataflow frame stays alive long enough.
    using FramePtr =
        hpx::intrusive_ptr<typename std::remove_pointer<typename std::decay<Frame>::type>::type>;
    FramePtr frame_p(frame);
    auto fn = [frame_p = std::move(frame_p), p = std::move(promise_next), comm = comm_,
               f = std::forward<F>(f), args = std::forward<TupleArgs>(args)](hpx::future<void>) mutable {
      MPI_Request req;
      auto all_args = hpx::tuple_cat(std::move(args), detail::extra_params<M>()(comm, &req));

      using result_t = decltype(hpx::util::invoke_fused(f, all_args));
      detail::invoke_fused_wrapper<result_t> wrapper(std::move(f), std::move(all_args));

      p.set_value();
      detail::nbmpi_internal_mech<M>{}(req);
      frame_p->set_data(wrapper.dataflow_return());
    };

    before_last.then(ex_, std::move(fn));
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
