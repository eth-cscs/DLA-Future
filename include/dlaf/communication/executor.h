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
#include <hpx/mutex.hpp>
#include <hpx/type_support/unused.hpp>

#include <mpi.h>

#include "dlaf/common/assert.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/init.h"
#include "dlaf/communication/mech.h"

namespace dlaf {
namespace comm {
namespace internal {

// Requests are only handled for the non-blocking version of comm::Executor
template <MPIMech mech>
struct request_handler {};

inline void handle_request(MPIMech mech, MPI_Request req) {
  if (mech == MPIMech::Yielding) {
    hpx::util::yield_while([&req] {
      int flag;
      mpi_invoke(MPI_Test, &req, &flag, MPI_STATUS_IGNORE);
      return flag == 0;
    });
  }
  else if (mech == MPIMech::Polling) {
    hpx::mpi::experimental::get_future(req).get();
  }
  else {
    std::cout << "UNIMPLEMENTED!" << std::endl;
    std::terminate();
  }
}

// Makes a tuple of the required additional arguments for blocking and non-blocking versions of comm::Executor
template <class Tuple>
auto make_mpi_tuple(Tuple t, MPI_Request* req_ptr) {
  return hpx::tuple_cat(std::move(t), hpx::make_tuple(req_ptr));
}

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

class Executor {
  hpx::execution::parallel_executor ex_;
  MPIMech mech_;

public:
  // Notes:
  //   - MPI event polling has to be enabled for `MPIMech::Polling`.
  Executor(const std::string& pool, MPIMech mech = MPIMech::Yielding)
      : ex_(&hpx::resource::get_thread_pool(pool)), mech_(mech) {}

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
    auto fn = [mech = mech_, f = std::forward<F>(f),
               args = hpx::make_tuple(std::forward<Ts>(ts)...)]() mutable {
      MPI_Request req;
      auto all_args = internal::make_mpi_tuple(std::move(args), &req);
      using result_t = decltype(hpx::util::invoke_fused(f, all_args));
      internal::invoke_fused_wrapper<result_t> wrapper(std::move(f), std::move(all_args));
      internal::handle_request(mech, req);
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
    auto fn = [mech = mech_, frame_p = std::move(frame_p), f = std::forward<F>(f),
               args = std::forward<TupleArgs>(args)]() mutable {
      MPI_Request req;
      auto all_args = internal::make_mpi_tuple(std::move(args), &req);
      using result_t = decltype(hpx::util::invoke_fused(f, all_args));
      internal::invoke_fused_wrapper<result_t> wrapper(std::move(f), std::move(all_args));
      internal::handle_request(mech, req);
      frame_p->set_data(wrapper.dataflow_return());
    };
    hpx::apply(ex_, std::move(fn));
  }
};
}
}

namespace hpx {
namespace parallel {
namespace execution {

template <>
struct is_two_way_executor<dlaf::comm::Executor> : std::true_type {};

}
}
}
