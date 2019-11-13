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

    Wrapper(U&& object) : object_(std::move(object)) {}

  public:
    Wrapper(Wrapper&& rhs) : object_(std::move(rhs.object_)) {
      promise_ = std::move(rhs.promise_);
    }

    ~Wrapper() {
      if (promise_)
        promise_->set_value(Wrapper<U>(std::move(object_)));
    }

    U& operator()() {
      return object_;
    }

    const U& operator()() const {
      return object_;
    }

  private:
    Wrapper<U>& set_promise(hpx::promise<Wrapper<U>>&& next_promise) {
      assert(!promise_);
      promise_ = std::make_unique<hpx::promise<Wrapper<U>>>(std::move(next_promise));
      return *this;
    }

    U object_;
    std::unique_ptr<hpx::promise<Wrapper<U>>> promise_;
  };

  Pipeline(T&& object) {
    future_ = hpx::make_ready_future(std::move(Wrapper<T>(std::move(object))));
  }

  ~Pipeline() {
    if (future_.valid())
      future_.get();
  }

  hpx::future<Wrapper<T>> operator()() {
    auto before_last = std::move(future_);

    hpx::promise<Wrapper<T>> promise;
    future_ = promise.get_future();

    return before_last.then(hpx::launch::sync, hpx::util::unwrapping(
                            [p = std::move(promise)](Wrapper<T>&& wrapper) mutable {
                              return std::move(wrapper.set_promise(std::move(p)));
                            }));
  }

private:
  hpx::future<Wrapper<T>> future_;
};

}
}
