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
/// It accepts both signed and unsigned integer types, but as precondition states both values must
/// not be negative (additionally, @p den must not be zero either)
///
/// @tparam IntType must be a signed or unsigned integer type.
/// @param num
/// @param den
/// @return ceiled division of type @a IntType
///
/// @pre @a num >= 0 and @a den > 0.
/// @pre @a num + @a den - 1 must not overflow @a IntType max value
template <typename IntType>
constexpr auto ceilDiv(const IntType num, const IntType den)
#ifdef DLAF_DOXYGEN
    ;
#else
    -> std::enable_if_t<std::is_integral<IntType>::value, IntType> {
  return (num + den - 1) / den;
}
#endif

}
}
