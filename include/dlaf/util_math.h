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

#include <type_traits>

/// @file

namespace dlaf {
namespace util {

/// @brief Returns ceiling(num/den) for integer types.
///
/// @tparam IntType has to be an integer type.
/// @param num
/// @param den
/// @return constexpr IntType
///
/// @pre @a num >= 0 and @a den >= 0.
template <class IntType>
constexpr std::enable_if_t<std::is_integral<IntType>::value, IntType> ceilDiv(IntType num, IntType den) {
  return (num + den - 1) / den;
}

}
}
