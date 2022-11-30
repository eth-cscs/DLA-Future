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

#include <pika/execution.hpp>

#include <functional>

namespace dlaf::common::internal {
/// Unwrapper is used to unwrap things. The default implementation does
/// nothing. It can be customized by specializing the Unwrapper struct.
template <typename T>
struct Unwrapper {
  template <typename U>
  static decltype(auto) unwrap(U&& u) {
    return std::forward<U>(u);
  }
};

// The external types future, shared_future, and reference_wrapper are
// unwrapped by calling their get methods.
template <typename T>
struct Unwrapper<pika::future<T>> {
  template <typename U>
  static decltype(auto) unwrap(U&& u) {
    static_assert(
        sizeof(T) == 0,
        "pika::futures should not be unwrapped automatically. You are most likely using keep_future on a future, in which case you should remove keep_future.");
  }
};

template <typename T>
struct Unwrapper<pika::shared_future<T>> {
  template <typename U>
  static decltype(auto) unwrap(U&& u) {
    return u.get();
  }
};

template <typename T>
struct Unwrapper<std::reference_wrapper<T>> {
  template <typename U>
  static decltype(auto) unwrap(U&& u) {
    return u.get();
  }
};

/// unwrap unwraps things in a way specified by Unwrapper.
template <typename T>
decltype(auto) unwrap(T&& t) {
  return Unwrapper<std::decay_t<T>>::unwrap(std::forward<T>(t));
}

/// Unwrapping is a callable object wrapper that calls unwrap on all arguments
/// before calling the wrapped callable.
template <typename F>
struct Unwrapping {
  std::decay_t<F> f;

  template <typename... Ts>
  auto operator()(Ts&&... ts)
      -> decltype(std::move(f)(Unwrapper<std::decay_t<Ts>>::unwrap(std::forward<Ts>(ts))...)) {
    return std::move(f)(Unwrapper<std::decay_t<Ts>>::unwrap(std::forward<Ts>(ts))...);
  }
};

template <typename F>
Unwrapping(F&&) -> Unwrapping<std::decay_t<F>>;
}
