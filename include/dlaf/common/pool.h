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

#include <functional>

#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/channel.hpp>

namespace dlaf {
namespace common {

template <typename T>
class Pool {
  template <class U>
  class Wrapper {
    friend class Pool<U>;

    Wrapper(U&& object, hpx::lcos::local::channel<U>* channel)
        : channel_(channel), object_(std::move(object)) {}

  public:
    Wrapper(Wrapper&& rhs) : channel_(rhs.channel_), object_(std::move(rhs.object_)) {
      rhs.channel_ = nullptr;
    }

    ~Wrapper() {
      if (channel_)
        channel_->set(std::move(object_));
    }

    U& operator()() {
      return object_;
    }

    const U& operator()() const {
      return object_;
    }

  private:
    U object_;
    hpx::lcos::local::channel<U>* channel_;
  };

public:
  using future_t = hpx::future<Wrapper<T>>;

  Pool(std::size_t pool_size) : size_(pool_size) {
    for (int i = 0; i < size_; ++i)
      channel_.set(T{});
  }

  ~Pool() {
    channel_.close(/*true*/);  // TODO check what force_delete does
    for (int i = 0; i < size_; ++i)
      channel_.get().get();
  }

  hpx::future<Wrapper<T>> get() {
    return channel_.get().then(hpx::launch::sync,
                               hpx::util::unwrapping(
                                   std::bind(&Pool::make_wrapper, this, std::placeholders::_1)));
  }

private:
  Wrapper<T> make_wrapper(T&& object) {
    return Wrapper<T>{std::move(object), &channel_};
  }

  const std::size_t size_;
  hpx::lcos::local::channel<T> channel_;
};

}
}
