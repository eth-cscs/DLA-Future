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

#include <type_traits>
#include <utility>

namespace dlaf::internal {
// Based on https://quuxplusone.github.io/blog/2018/05/17/super-elider-round-2/ and
// https://akrzemi1.wordpress.com/2018/05/16/rvalues-redefined/.
//
// Because of the conversion operator and guaranteed copy-elision, useful for
// emplacing immovable types into e.g. variants and optionals. Can also be used
// to construct new instances for each element in a vector, where the element
// type has reference semantics and regular copy construction is not what is
// wanted.
template <typename F>
class WithResultOf {
  F&& f;

public:
  using ResultType = std::invoke_result_t<F&&>;
  explicit WithResultOf(F&& f) : f(std::forward<F>(f)) {}
  operator ResultType() {
    return std::forward<F>(f)();
  }
};

template <typename F>
WithResultOf(F&&) -> WithResultOf<F&&>;
}
