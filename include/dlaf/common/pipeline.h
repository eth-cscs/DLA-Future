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

#include <pika/future.hpp>
#include <pika/unwrap.hpp>

namespace dlaf {
namespace common {

/// A `promise` like type which is set upon destruction. The type separates the placement of data
/// (T) into the promise from notifying the corresponding `pika::future`.
///
/// Note: The type is non-copiable.
template <class T>
class PromiseGuard {
public:
  /// Create a wrapper.
  /// @param object the resource to wrap (the wrapper becomes the owner of the resource),
  /// @param next the promise that has to be set on destruction.
  PromiseGuard(T object, pika::lcos::local::promise<T> next)
      : object_(std::move(object)), promise_(std::move(next)) {}

  PromiseGuard(PromiseGuard&&) = default;
  PromiseGuard& operator=(PromiseGuard&&) = default;

  PromiseGuard(const PromiseGuard&) = delete;
  PromiseGuard& operator=(const PromiseGuard&) = delete;

  /// This is where the "magic" happens!
  ///
  /// If the wrapper is still valid, set the promise to unlock the next future.
  ~PromiseGuard() {
    if (promise_.valid())
      promise_.set_value(std::move(object_));
  }

  /// Get a reference to the internal object.
  T& ref() {
    return object_;
  }

  const T& ref() const {
    return object_;
  };

private:
  T object_;                               /// the object owned by the wrapper.
  pika::lcos::local::promise<T> promise_;  /// the shared state that will unlock the next user.
};

/// Pipeline takes ownership of a given object and manages the access to this resource by serializing
/// calls. Anyone that requires access to the underlying resource will get an pika::future, which is the
/// way to register to the queue. All requests are serialized and served in the same order they arrive.
/// On destruction it does not wait for the queued requests for the resource and exits immediately.
///
/// The mechanism for auto-releasing the resource and passing it to the next user works thanks to the
/// internal PromiseGuard object. This PromiseGuard contains the real resource, and it will do what is
/// needed to unlock the next user as soon as the PromiseGuard is destroyed.
template <class T>
class Pipeline {
public:
  /// Create a Pipeline by moving in the resource (it takes the ownership).
  Pipeline(T object) {
    future_ = pika::make_ready_future(std::move(object));
  }

  /// Enqueue for the resource.
  ///
  /// @return a future that will become ready as soon as the previous user release the resource.
  pika::future<PromiseGuard<T>> operator()() {
    auto before_last = std::move(future_);

    pika::lcos::local::promise<T> promise_next;
    future_ = promise_next.get_future();

    return before_last.then(pika::launch::sync,
                            pika::unwrapping([promise_next = std::move(promise_next)](T object) mutable {
                              return PromiseGuard<T>{std::move(object), std::move(promise_next)};
                            }));
  }

private:
  pika::future<T> future_;  ///< This contains always the "tail" of the queue of futures.
};
}
}
