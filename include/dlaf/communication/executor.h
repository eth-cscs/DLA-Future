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
  template <class Tuple>
  static auto call(Tuple t, MPI_Request* req_ptr) {
    return hpx::tuple_cat(std::move(t), hpx::make_tuple(req_ptr));
  }
};

template <>
struct make_mpi_tuple<MPIMech::Blocking> {
  template <class Tuple>
  static auto call(Tuple t, MPI_Request*) {
    return std::move(t);
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
  detail::hint_manager_wrapper<M> mgr_;
  hpx::execution::parallel_executor ex_;

public:
  // Notes:
  //   - MPI event polling has to be enabled for `MPIMech::Polling`.
  Executor(const std::string& pool) {
    detail::init_parallel_executor_with_hint(pool, mgr_, ex_);
  }

  bool operator==(const Executor& rhs) const noexcept {
    return ex_ == rhs.ex_;
  }

  bool operator!=(const Executor& rhs) const noexcept {
    return !(*this == rhs);
  }

  const Executor& context() const noexcept {
    return *this;
  }

  template <typename F, typename... Ts>
  auto async_execute(F&& f, Ts&&... ts) noexcept {
    auto fn = [f = std::forward<F>(f), args = hpx::make_tuple(std::forward<Ts>(ts)...)]() mutable {
      MPI_Request req;
      auto all_args = detail::make_mpi_tuple<M>::call(std::move(args), &req);
      using result_t = decltype(hpx::util::invoke_fused(f, all_args));
      detail::invoke_fused_wrapper<result_t> wrapper(std::move(f), std::move(all_args));
      detail::request_handler<M>::call(req);
      return wrapper.async_return();
    };
    return hpx::async(ex_, std::move(fn));
  }

  template <class Frame, class F, class TupleArgs>
  void dataflow_finalize(Frame&& frame, F&& f, TupleArgs&& args) {
    // Ensure the dataflow frame stays alive long enough.
    using FramePtr =
        hpx::intrusive_ptr<typename std::remove_pointer<typename std::decay<Frame>::type>::type>;
    FramePtr frame_p(std::forward<Frame>(frame));
    auto fn = [frame_p = std::move(frame_p), f = std::forward<F>(f),
               args = std::forward<TupleArgs>(args)]() mutable {
      MPI_Request req;
      auto all_args = detail::make_mpi_tuple<M>::call(std::move(args), &req);
      using result_t = decltype(hpx::util::invoke_fused(f, all_args));
      detail::invoke_fused_wrapper<result_t> wrapper(std::move(f), std::move(all_args));
      detail::request_handler<M>::call(req);
      frame_p->set_data(wrapper.dataflow_return());
    };
    hpx::async(ex_, std::move(fn));
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
