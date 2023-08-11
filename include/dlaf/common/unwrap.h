//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

/// @file

#include <functional>
#include <type_traits>

#include <pika/async_rw_mutex.hpp>
#include <pika/execution.hpp>

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

template <typename T>
struct Unwrapper<std::reference_wrapper<T>> {
  template <typename U>
  static decltype(auto) unwrap(U&& u) {
    return u.get();
  }
};

template <typename T1, typename T2, pika::execution::experimental::async_rw_mutex_access_type at>
struct Unwrapper<pika::execution::experimental::async_rw_mutex_access_wrapper<T1, T2, at>> {
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
