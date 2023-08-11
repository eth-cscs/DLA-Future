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
/// ConsumeRvalues is a callable object wrapper that consumes rvalues passed as arguments
/// after calling the wrapped callable.
template <typename F>
struct ConsumeRvalues {
  std::decay_t<F> f;

  template <typename... Ts>
  auto operator()(Ts&&... ts) -> decltype(std::move(f)(std::forward<Ts>(ts)...)) {
    using result_type = decltype(std::move(f)(std::forward<Ts>(ts)...));
    if constexpr (std::is_void_v<result_type>) {
      std::move(f)(std::forward<Ts>(ts)...);
      std::tuple<Ts...>(std::forward<Ts>(ts)...);
    }
    else {
      auto r = std::move(f)(std::forward<Ts>(ts)...);
      std::tuple<Ts...>(std::forward<Ts>(ts)...);
      return r;
    }
  }
};

template <typename F>
ConsumeRvalues(F&&) -> ConsumeRvalues<std::decay_t<F>>;

}
