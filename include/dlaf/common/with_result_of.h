//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

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
class with_result_of {
  F&& f;

public:
  using result_type = std::invoke_result_t<F&&>;
  explicit with_result_of(F&& f) : f(std::forward<F>(f)) {}
  operator result_type() {
    return std::forward<F>(f)();
  }
};

template <typename F>
with_result_of(F&&) -> with_result_of<F&&>;
}
