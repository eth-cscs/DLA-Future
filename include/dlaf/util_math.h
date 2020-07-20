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

#include <cstddef>
#include <functional>
#include <type_traits>

#include "dlaf/types.h"

/// @file

namespace dlaf {
namespace util {

/// Returns ceiling(num/den) for integer types.
///
/// It accepts both signed and unsigned integer types, but as precondition states both values must
/// not be negative (additionally, @p den must not be zero either).
///
/// @tparam IntType must be a signed or unsigned integer type,
/// @param num,
/// @param den,
/// @return ceiled division of type @a IntType.
///
/// @pre @a num >= 0 and @a den > 0,
/// @pre @a num + @a den - 1 must not overflow @a IntType max value.
#ifdef DLAF_DOXYGEN
template <typename IntType>
constexpr IntType ceilDiv(const IntType num, const IntType den);
#else
template <typename IntType>
constexpr auto ceilDiv(const IntType num, const IntType den)
    -> std::enable_if_t<std::is_integral<IntType>::value, IntType> {
  return (num + den - 1) / den;
}
#endif

namespace internal {

/// Perform the given binary operation on integer types in @tparam ArithmeticT arithmetic.
///
/// It casts operands to @tparam ArithmeticT type and then performs the given binary operation @p op.
/// @tparam TA and @tparam TB must be integer types (signed or unsigned),
/// @return @p a @p op @p b.
/// @pre @p a and @p b can be stored in the type @tparam ArithmeticT,
/// @pre it must be possible to store the result in @tparam ArithmeticT.
template <class ArithmeticT, class TA, class TB, class BinaryOp>
constexpr ArithmeticT generic_integer_op(const TA a, const TB b, BinaryOp op) {
  return op(integral_cast<ArithmeticT>(a), integral_cast<ArithmeticT>(b));
}

}

namespace size_t {

/// Perform the sum on integer types in size_t arithmetic.
///
/// It casts operands to std::size_t type and then performs the sum.
/// @tparam TA and @tparam TB must be integer types (signed or unsigned),
/// @return @p a + @p b.
/// @pre @p a >= 0,
/// @pre @p b >= 0,
/// @pre it must be possible to store the result in std::size_t.
template <typename TA, typename TB>
std::size_t sum(const TA a, const TB b) {
  return dlaf::util::internal::generic_integer_op<std::size_t>(a, b, std::plus<std::size_t>());
}

/// Perform the multiplication on integer types in size_t arithmetic.
///
/// It casts operands to std::size_t type and then performs the multiplication.
/// @tparam TA and @tparam TB must be integer types (signed or unsigned),
/// @return @p a * @p b.
/// @pre @p a >= 0,
/// @pre @p b >= 0,
/// @pre it must be possible to store the result in std::size_t.
template <typename TA, typename TB>
std::size_t mul(const TA a, const TB b) {
  return dlaf::util::internal::generic_integer_op<std::size_t>(a, b, std::multiplies<std::size_t>());
}

}

namespace ptrdiff_t {

/// Perform the sum on integer types in ptrdiff_t arithmetic.
///
/// It casts operands to std::ptrdiff_t type and then performs the sum.
/// @tparam TA and @tparam TB must be integer types (signed or unsigned),
/// @return @p a + @p b.
/// @pre @p a and @p b can be stored in std::ptrdiff_t,
/// @pre it must be possible to store the result in std::ptrdiff_t.
template <typename TA, typename TB>
std::ptrdiff_t sum(const TA a, const TB b) {
  return dlaf::util::internal::generic_integer_op<std::ptrdiff_t>(a, b, std::plus<std::ptrdiff_t>());
}

/// Perform the multiplication on integer types in ptrdiff_t arithmetic.
///
/// It casts operands to std::ptrdiff_t type and then performs the multiplication.
/// @tparam TA and @tparam TB must be integer types (signed or unsigned),
/// @return @p a * @p b.
/// @pre @p a and @p b can be stored in std::ptrdiff_t,
/// @pre it must be possible to store the result in std::ptrdiff_t.
template <typename TA, typename TB>
std::ptrdiff_t mul(const TA a, const TB b) {
  return dlaf::util::internal::generic_integer_op<std::ptrdiff_t>(a, b,
                                                                  std::multiplies<std::ptrdiff_t>());
}

}
}
}
