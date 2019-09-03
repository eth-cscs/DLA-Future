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

/// @file

namespace dlaf {
namespace util {

/// @brief Returns ceiling(num/den) for integer types.
///
/// It accepts both signed and unsigned integer types, but as precondition states both values must
/// not be negative (additionally, @p den must not be zero either)
///
/// @tparam IntNumType and @tparam IntDenType have to be integer types.
/// @param num
/// @param den
/// @return ceiled division of type @a IntNumType
///
/// @pre @a num >= 0 and @a den > 0.
/// @pre @a num + @a den - 1 must not overflow @a IntNumType max value
#ifndef DLAF_DOXYGEN
template <typename IntNumType, typename IntDenType>
constexpr IntNumType ceilDiv(const IntNumType num, const IntDenType den);
#else
template <typename IntNumType, typename IntDenType>
constexpr
std::enable_if_t<std::is_integral<IntNumType>::value && std::is_integral<IntDenType>::value, IntNumType>
ceilDiv(const IntNumType num, const IntDenType den) {
  return (num + den - 1) / den;
}
#endif

}
}
