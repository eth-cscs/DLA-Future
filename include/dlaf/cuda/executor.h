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

#ifdef DLAF_WITH_CUDA

#include <cstddef>
#include <memory>
#include <utility>

#include <cuda_runtime.h>

#include <hpx/local/execution.hpp>
#include <hpx/local/functional.hpp>
#include <hpx/local/future.hpp>
#include <hpx/local/tuple.hpp>
#include <hpx/local/unwrap.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/cuda/error.h"
#include "dlaf/cuda/stream_pool.h"

namespace dlaf {
namespace cuda {

/// An executor for CUDA functions. Uses streams from the given StreamPool. A
/// CUDA function is defined as any function that takes a CUDA stream as the
/// first argument. The executor inserts a CUDA stream into the argument list,
/// i.e. a stream should not be provided at the apply/async/dataflow invocation
/// site.
class Executor {
protected:
  StreamPool stream_pool_;

public:
  Executor(StreamPool stream_pool) : stream_pool_(stream_pool) {}

  bool operator==(Executor const& rhs) const noexcept {
    return stream_pool_ == rhs.stream_pool_;
  }

  bool operator!=(Executor const& rhs) const noexcept {
    return !(*this == rhs);
  }

  Executor const& context() const noexcept {
    return *this;
  }

  template <typename F, typename... Ts>
  auto async_execute(F&& f, Ts&&... ts) {
    cudaStream_t stream = stream_pool_.getNextStream();
    auto r = hpx::invoke(std::forward<F>(f), std::forward<Ts>(ts)..., stream);
    hpx::future<void> fut = hpx::cuda::experimental::detail::get_future_with_event(stream);

    // The stream pool is captured by value to ensure that the streams live at
    // least until the event has completed.
    return fut.then(hpx::launch::sync, [r = std::move(r), stream_pool = stream_pool_](
                                           hpx::future<void>&&) mutable { return std::move(r); });
  }

  template <class Frame, class F, class Futures>
  void dataflow_finalize(Frame&& frame, F&& f, Futures&& futures) {
    // Ensure the dataflow frame stays alive long enough.
    hpx::intrusive_ptr<typename std::remove_pointer<typename std::decay<Frame>::type>::type> frame_p(
        frame);

    cudaStream_t stream = stream_pool_.getNextStream();
    auto r = hpx::invoke_fused(std::forward<F>(f),
                               hpx::tuple_cat(std::forward<Futures>(futures), hpx::tie(stream)));
    hpx::future<void> fut = hpx::cuda::experimental::detail::get_future_with_event(stream);

    // The stream pool is captured by value to ensure that the streams live at
    // least until the event has completed.
    fut.then(hpx::launch::sync,
             [r = std::move(r), frame_p = std::move(frame_p), stream_pool = stream_pool_](
                 hpx::future<void>&&) mutable { frame_p->set_data(std::move(r)); });
  }
};
}
}

namespace hpx {
namespace parallel {
namespace execution {

template <>
struct is_two_way_executor<dlaf::cuda::Executor> : std::true_type {};
}
}
}

#endif
