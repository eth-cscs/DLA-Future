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

#include <hpx/include/util.hpp>
#include <hpx/local/future.hpp>

namespace dlaf {
namespace common {

/// Pipeline takes ownership of a given object and manages the access to this resource by serializing
/// calls. Anyone that requires access to the underlying resource will get an hpx::future, which is the
/// way to register to the queue. All requests are serialized and served in the same order they arrive.
///
/// The mechanism for auto-releasing the resource and passing it to the next user works thanks to the
/// internal Wrapper object. This Wrapper contains the real resource, and it will do what is needed to
/// unlock the next user as soon as the Wrapper is destroyed.
template <class T>
class Pipeline {
  template <class PT>
  using promise_t = hpx::lcos::local::promise<PT>;

public:
  /// Wrapper is the object that manages the auto-release mechanism.
  template <class U>
  class Wrapper {
    friend class Pipeline<U>;

    /// Create a wrapper.
    /// @param object	the resource to wrap (the wrapper becomes the owner of the resource),
    /// @param next	the promise that has to be set on destruction.
    Wrapper(U&& object, promise_t<T> next) : object_(std::move(object)), promise_(std::move(next)) {}

  public:
    /// Trivial move constructor (that invalidates the status of the source object).
    Wrapper(Wrapper&& rhs) = default;

    /// This is where the "magic" happens!
    ///
    /// If the wrapper is still valid, set the promise to unlock the next future.
    ~Wrapper() {
      if (promise_.valid())
        promise_.set_value(std::move(object_));
    }

    /// Get a reference to the internal object.
    U& operator()() {
      return object_;
    }

    /// Get a reference to the internal object.
    const U& operator()() const {
      return object_;
    }

  private:
    U object_;  ///< the wrapped object! it is actually owned by the wrapper.
    /// promise containing the shared state that will unlock the next user.
    promise_t<U> promise_;
  };

  /// Create a Pipeline by moving in the resource (it takes the ownership).
  Pipeline(T&& object) {
    future_ = hpx::make_ready_future(std::move(object));
  }

  /// On destruction it waits that all users have finished using it.
  ~Pipeline() {
    if (future_.valid())
      future_.get();
  }

  /// Enqueue for the resource.
  ///
  /// @return a future that will become ready as soon as the previous user release the resource.
  hpx::future<Wrapper<T>> operator()() {
    auto before_last = std::move(future_);

    promise_t<T> promise_next;
    future_ = promise_next.get_future();

    return before_last.then(hpx::launch::sync,
                            hpx::util::unwrapping(
                                [promise_next = std::move(promise_next)](T&& object) mutable {
                                  return Wrapper<T>{std::move(object), std::move(promise_next)};
                                }));
  }

private:
  hpx::future<T> future_;  ///< This contains always the "tail" of the queue of futures.
};
}
}
