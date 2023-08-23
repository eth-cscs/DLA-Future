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

#include <tuple>
#include <type_traits>
#include <utility>

namespace dlaf::common::internal {

template <typename T>
T consume_rvalue(T&& x) {
  return std::move(x);
}

template <typename T>
T& consume_rvalue(T& x) {
  return x;
}

/// ConsumeRvalues is a callable object wrapper that consumes rvalues passed as arguments
/// after calling the wrapped callable.
template <typename F>
struct ConsumeRvalues {
  std::decay_t<F> f;

  template <typename... Ts>
  auto operator()(Ts&&... ts) -> decltype(std::move(f)(consume_rvalue(std::forward<Ts>(ts))...)) {
    return std::move(f)(consume_rvalue(std::forward<Ts>(ts))...);
  }
};

template <typename F>
ConsumeRvalues(F&&) -> ConsumeRvalues<std::decay_t<F>>;

}
