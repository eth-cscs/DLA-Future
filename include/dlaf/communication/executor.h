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
struct request_handler {};

template <>
struct request_handler<MPIMech::Polling> {
  static void call(MPI_Request req) {
    hpx::mpi::experimental::get_future(req).get();
  }
};

template <>
struct request_handler<MPIMech::Yielding> {
  static void call(MPI_Request req) {
    hpx::util::yield_while([&req] {
      int flag;
      mpi_invoke(MPI_Test, &req, &flag, MPI_STATUS_IGNORE);
      return flag == 0;
    });
  }
};

template <>
struct request_handler<MPIMech::Blocking> {
  static void call(MPI_Request) {}
};

// Makes a tuple of the required additional arguments for blocking and non-blocking versions of comm::Executor
template <MPIMech M>
struct make_mpi_tuple {
  static hpx::tuple<Communicator, MPI_Request*> call(Communicator comm, MPI_Request* req_ptr) {
    return hpx::make_tuple(std::move(comm), req_ptr);
  }
};

template <>
struct make_mpi_tuple<MPIMech::Blocking> {
  static hpx::tuple<Communicator> call(Communicator comm, MPI_Request*) {
    return hpx::make_tuple(std::move(comm));
  }
};

// Wraps the invocation of `F` such that the void and non-void cases are handled without code duplication.
template <class R>
struct invoke_fused_wrapper {
  R val;
  template <class F, class TupleArgs>
  invoke_fused_wrapper(F&& f, TupleArgs&& ts)
      : val(hpx::invoke_fused(std::forward<F>(f), std::forward<TupleArgs>(ts))) {}
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
  class TaskChain;  // forward declaration

  Communicator comm_;
  std::shared_ptr<TaskChain> tc_ptr;
  detail::hint_manager_wrapper<M> mgr_;
  hpx::execution::parallel_executor ex_;

  class TaskChain {
    hpx::future<void> tail_;
    hpx::lcos::local::mutex mt_;

  public:
    TaskChain() : tail_(hpx::make_ready_future<void>()) {}

    void chain(hpx::future<void>& before_last, hpx::lcos::local::promise<void>& promise_next) {
      std::lock_guard<hpx::lcos::local::mutex> lk(mt_);
      before_last = std::move(tail_);
      tail_ = promise_next.get_future();
    }
  };

  struct task_chain_deleter {
    void operator()(TaskChain* /*tc*/) {
      // if (tc->tail.valid()) {
      //  tc->tail.get();
      //}
    }
  };

public:
  // TODO: REMOVE! THIS IS ONLY FOR DEBUGGING
  std::string msg;

  // Notes:
  //   - MPI event polling has to be enabled for `MPIMech::Polling`.
  Executor(const std::string& pool, Communicator comm)
      : comm_(std::move(comm)),
        tc_ptr(std::shared_ptr<TaskChain>(new TaskChain(), task_chain_deleter())), msg("") {
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
    tc_ptr->chain(before_last, promise_next);

    auto fn = [msg = msg, p = std::move(promise_next), comm = comm_, f = std::forward<F>(f),
               args = hpx::make_tuple(std::forward<Ts>(ts)...)](hpx::future<void>) mutable {
      MPI_Request req;
      auto all_args =
          hpx::tuple_cat(std::move(args), detail::make_mpi_tuple<M>::call(std::move(comm), &req));
      using result_t = decltype(hpx::util::invoke_fused(f, all_args));
      detail::invoke_fused_wrapper<result_t> wrapper(std::move(f), std::move(all_args));
      p.set_value();
      detail::request_handler<M>::call(req);
      std::cout << "ASYNC : " << msg << std::endl;
      return wrapper.async_return();
    };
    return before_last.then(ex_, std::move(fn));
  }

  template <class Frame, class F, class TupleArgs>
  void dataflow_finalize(Frame&& frame, F&& f, TupleArgs&& args) {
    hpx::future<void> before_last;
    hpx::lcos::local::promise<void> promise_next;
    tc_ptr->chain(before_last, promise_next);

    // Ensure the dataflow frame stays alive long enough.
    using FramePtr =
        hpx::intrusive_ptr<typename std::remove_pointer<typename std::decay<Frame>::type>::type>;
    FramePtr frame_p(std::forward<Frame>(frame));
    auto fn = [msg = msg, frame_p = std::move(frame_p), p = std::move(promise_next), comm = comm_,
               f = std::forward<F>(f), args = std::forward<TupleArgs>(args)](hpx::future<void>) mutable {
      MPI_Request req;
      auto all_args =
          hpx::tuple_cat(std::move(args), detail::make_mpi_tuple<M>::call(std::move(comm), &req));
      using result_t = decltype(hpx::util::invoke_fused(f, all_args));
      detail::invoke_fused_wrapper<result_t> wrapper(std::move(f), std::move(all_args));
      p.set_value();
      detail::request_handler<M>::call(req);
      std::cout << "DATAFLOW : " << msg << std::endl;
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
