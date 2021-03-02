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
#include <mutex>
#include <type_traits>
#include <utility>

#include <hpx/async_mpi/mpi_future.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution.hpp>
#include <hpx/functional.hpp>
#include <hpx/future.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/synchronization/mutex.hpp>
#include <hpx/type_support/unused.hpp>

#include <mpi.h>

#include "dlaf/common/assert.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/hints.h"
#include "dlaf/communication/init.h"
#include "dlaf/communication/mech.h"

namespace dlaf {
namespace comm {
namespace detail {

// Hints are only used for the blocking version of comm::Executor
template <MPIMech M>
struct hint_manager_wrapper {};

template <>
struct hint_manager_wrapper<MPIMech::Blocking> {
  hint_manager mgr;
};

template <MPIMech M>
void init_parallel_executor_with_hint(const std::string& pool, hint_manager_wrapper<M>& mgr_wrapper,
                                      hpx::execution::parallel_executor& ex) {
  ex = hpx::execution::parallel_executor(&hpx::resource::get_thread_pool(pool));
  mgr_wrapper = hint_manager_wrapper<M>();
}

inline void init_parallel_executor_with_hint(const std::string& pool,
                                             hint_manager_wrapper<MPIMech::Blocking>& mgr_wrapper,
                                             hpx::execution::parallel_executor& ex) {
  mgr_wrapper.mgr = hint_manager(pool);
  ex = hpx::execution::parallel_executor(&hpx::resource::get_thread_pool(pool),
                                         hpx::threads::thread_priority::default_,
                                         hpx::threads::thread_stacksize::nostack,
                                         hpx::threads::thread_schedule_hint(
                                             mgr_wrapper.mgr.get_thread_index()));
}

// Requests are only handled for the non-blocking version of comm::Executor
template <MPIMech mech>
struct request_handler_fn {};

template <>
struct request_handler_fn<MPIMech::Polling> {
  void operator()(MPI_Request req) {
    hpx::mpi::experimental::get_future(req).get();
  }
};

template <>
struct request_handler_fn<MPIMech::Yielding> {
  void operator()(MPI_Request req) {
    hpx::util::yield_while([&req] {
      int flag;
      mpi_invoke(MPI_Test, &req, &flag, MPI_STATUS_IGNORE);
      return flag == 0;
    });
  }
};

template <>
struct request_handler_fn<MPIMech::Blocking> {
  void operator()(MPI_Request) {}
};

// Makes a tuple of the required additional arguments for blocking and non-blocking versions of comm::Executor
template <MPIMech M>
struct make_tuple_from_mpi_data_fn {
  hpx::tuple<Communicator, MPI_Request*> operator()(Communicator comm, MPI_Request* req_ptr) {
    return hpx::make_tuple(std::move(comm), req_ptr);
  }
};

template <>
struct make_tuple_from_mpi_data_fn<MPIMech::Blocking> {
  hpx::tuple<Communicator> operator()(Communicator comm, MPI_Request*) {
    return hpx::make_tuple(std::move(comm));
  }
};

// Wraps the invocation of `F` such that the void and non-void cases are handled without code duplication.
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

}

template <MPIMech M>
class Executor {
  struct task_chain;  // forward declaration

  Communicator comm_;
  std::shared_ptr<task_chain> tc_ptr;
  detail::hint_manager_wrapper<M> mgr_;
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
  //   - MPI event polling has to be enabled for `MPIMech::Polling`.
  Executor(const std::string& pool, Communicator comm)
      : comm_(std::move(comm)),
        tc_ptr(std::shared_ptr<task_chain>(new task_chain(), task_chain_deleter())) {
    detail::init_parallel_executor_with_hint(pool, mgr_, ex_);
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
    hpx::future<void> before_last;
    hpx::lcos::local::promise<void> promise_next;
    {
      std::lock_guard<hpx::lcos::local::mutex> lk(tc_ptr->mt);
      before_last = std::move(tc_ptr->tail);
      tc_ptr->tail = promise_next.get_future();
    }
    auto fn = [p = std::move(promise_next), comm = comm_, f = std::forward<F>(f),
               args = hpx::make_tuple(std::forward<Ts>(ts)...)](hpx::future<void>) mutable {
      MPI_Request req;
      auto all_args =
          hpx::tuple_cat(std::move(args), detail::make_tuple_from_mpi_data_fn<M>()(comm, &req));
      using result_t = decltype(hpx::util::invoke_fused(f, all_args));
      detail::invoke_fused_wrapper<result_t> wrapper(std::move(f), std::move(all_args));
      p.set_value();
      detail::request_handler_fn<M>{}(req);
      return wrapper.async_return();
    };
    return before_last.then(ex_, std::move(fn));
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
      auto all_args =
          hpx::tuple_cat(std::move(args), detail::make_tuple_from_mpi_data_fn<M>()(comm, &req));
      using result_t = decltype(hpx::util::invoke_fused(f, all_args));
      detail::invoke_fused_wrapper<result_t> wrapper(std::move(f), std::move(all_args));
      p.set_value();
      detail::request_handler_fn<M>{}(req);
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
