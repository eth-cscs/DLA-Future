//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
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

#include <pika/async_mpi/mpi_future.hpp>
#include <pika/execution.hpp>
#include <pika/functional.hpp>
#include <pika/future.hpp>
#include <pika/mutex.hpp>
#include <pika/tuple.hpp>
#include <pika/type_support/unused.hpp>

#include <mpi.h>

#include "dlaf/common/assert.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/init.h"
#include "dlaf/communication/mech.h"
#include "dlaf/init.h"

namespace dlaf {
namespace comm {
namespace internal {

// Requests are only handled for the non-blocking version of comm::Executor
template <MPIMech mech>
struct request_handler {};

inline void handle_request(MPI_Request req) {
  MPIMech mech = dlaf::internal::getConfiguration().mpi_mech;
  if (mech == MPIMech::Yielding) {
    pika::util::yield_while([&req] {
      int flag;
      mpi_invoke(MPI_Test, &req, &flag, MPI_STATUS_IGNORE);
      return flag == 0;
    });
  }
  else if (mech == MPIMech::Polling) {
    pika::mpi::experimental::get_future(req).get();
  }
  else {
    std::cout << "UNIMPLEMENTED!" << std::endl;
    std::terminate();
  }
}

// Makes a tuple of the required additional arguments for blocking and non-blocking versions of comm::Executor
template <class Tuple>
auto make_mpi_tuple(Tuple t, MPI_Request* req_ptr) {
  return pika::tuple_cat(std::move(t), pika::make_tuple(req_ptr));
}

// Wraps the invocation of `F` such that the void and non-void cases are handled without code duplication.
template <class R>
struct invoke_fused_wrapper {
  R val;
  template <class F, class TupleArgs>
  invoke_fused_wrapper(F&& f, TupleArgs&& ts)
      : val(pika::invoke_fused(std::forward<F>(f), std::forward<TupleArgs>(ts))) {}
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
    pika::invoke_fused(std::forward<F>(f), std::forward<TupleArgs>(ts));
  }
  void async_return() {}
  auto dataflow_return() {
    return pika::util::unused;
  }
};

}

class Executor {
  pika::execution::parallel_executor ex_;

public:
  // Notes:
  //   - MPI event polling has to be enabled for `MPIMech::Polling`.
  Executor() : ex_(&pika::resource::get_thread_pool(dlaf::internal::getConfiguration().mpi_pool)) {}

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
    auto fn = [f = std::forward<F>(f), args = pika::make_tuple(std::forward<Ts>(ts)...)]() mutable {
      MPI_Request req;
      auto all_args = internal::make_mpi_tuple(std::move(args), &req);
      using result_t = decltype(pika::util::invoke_fused(f, std::move(all_args)));
      internal::invoke_fused_wrapper<result_t> wrapper(std::move(f), std::move(all_args));
      internal::handle_request(req);
      return wrapper.async_return();
    };
    return pika::async(ex_, std::move(fn));
  }

  template <class Frame, class F, class TupleArgs>
  void dataflow_finalize(Frame&& frame, F&& f, TupleArgs&& args) {
    // Ensure the dataflow frame stays alive long enough.
    using FramePtr =
        pika::intrusive_ptr<typename std::remove_pointer<typename std::decay<Frame>::type>::type>;
    FramePtr frame_p(std::forward<Frame>(frame));

    char const* annotation = pika::traits::get_function_annotation<F>::call(f);
    auto fn = [frame_p = std::move(frame_p), f = std::forward<F>(f),
               args = std::forward<TupleArgs>(args)]() mutable {
      MPI_Request req;
      auto all_args = internal::make_mpi_tuple(std::move(args), &req);
      using result_t = decltype(pika::util::invoke_fused(f, std::move(all_args)));
      internal::invoke_fused_wrapper<result_t> wrapper(std::move(f), std::move(all_args));
      internal::handle_request(req);
      frame_p->set_data(wrapper.dataflow_return());
    };
    pika::apply(ex_, pika::annotated_function(std::move(fn), annotation));
  }
};
}
}

namespace pika {
namespace parallel {
namespace execution {

template <>
struct is_two_way_executor<dlaf::comm::Executor> : std::true_type {};

}
}
}
