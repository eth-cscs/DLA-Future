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
#include <hpx/lcos/promise.hpp>

#include "dlaf/common/iwrapper.hpp"

namespace dlaf {
namespace common {

template <typename T>
class Pool {
public:
  using future_t = hpx::future<IWrapper<T>>;

  Pool(std::size_t pool_size) : size_(pool_size) {
    for (int i = 0; i < size_; ++i)
      channel_.set(T{});
  }

  ~Pool() {
    channel_.close(/*true*/);  // TODO check what force_delete does
    for (int i = 0; i < size_; ++i)
      channel_.get().get();
  }

  future_t get() {
    using hpx::util::unwrapping;

    return channel_.get().then(hpx::launch::sync, unwrapping([&channel = channel_](auto&& object) {
                                 auto wrapper = IWrapper<T>(std::move(object));
                                 hpx::promise<T> next_promise;
                                 next_promise.get_future().then(hpx::launch::sync,
                                                                unwrapping([&channel](auto&& object) {
                                                                  channel.set(std::move(object));
                                                                }));
                                 wrapper.setPromise(std::move(next_promise));
                                 return std::move(wrapper);
                               }));
  }

private:
  const std::size_t size_;
  hpx::lcos::local::channel<T> channel_;
};

}
}
