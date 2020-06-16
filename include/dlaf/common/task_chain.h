//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <hpx/lcos/future.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/lcos_fwd.hpp>

namespace dlaf {
namespace common {

class TaskChain {
  hpx::future<void> tail_;  // the tail of the chain

public:
  inline TaskChain() : tail_(hpx::make_ready_future()) {}

  /// Chain the task
  /// @return a future that will become ready as soon as the promise is set
  inline hpx::future<hpx::promise<void>> chain() {
    auto prev_tail = std::move(tail_);
    hpx::promise<void> promise_next;
    tail_ = promise_next.get_future();

    return prev_tail.then(hpx::launch::sync, [promise_next = std::move(promise_next)](auto&&) mutable {
      return std::move(promise_next);
    });
  }

  /// Wait until all tasks in the chain completed
  ~TaskChain() {
    if (tail_.valid())
      tail_.get();
  }
};

}
}
