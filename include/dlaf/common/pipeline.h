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

namespace dlaf {
namespace common {

template <class T>
struct Pipeline {
public:
  template <class U>
  class Wrapper {
    friend class Pipeline<U>;

    Wrapper(U&& object, hpx::promise<T> next)
      : object_(std::move(object)), promise_(std::move(next)), valid_(true) {}

  public:
    Wrapper(Wrapper&& rhs)
      : object_(std::move(rhs.object_)), promise_(std::move(rhs.promise_)) {
        std::swap(valid_, rhs.valid_);
      }

    ~Wrapper() {
      if (valid_)
        promise_.set_value(std::move(object_));
    }

    U& operator()() {
      return object_;
    }

    const U& operator()() const {
      return object_;
    }

  private:
    bool valid_ = false;
    U object_;
    hpx::promise<U> promise_;
  };

  Pipeline(T&& object) {
    future_ = hpx::make_ready_future(std::move(object));
  }

  ~Pipeline() {
    if (future_.valid())
      future_.get();
  }

  hpx::future<Wrapper<T>> operator()() {
    auto before_last = std::move(future_);

    hpx::promise<T> promise_next;
    future_ = promise_next.get_future();

    return before_last.then(hpx::launch::sync, hpx::util::unwrapping(
                            [promise_next = std::move(promise_next)](T&& object) mutable {
                              return Wrapper<T>{std::move(object), std::move(promise_next)};
                            }));
  }

private:
  hpx::future<T> future_;
};

}
}
