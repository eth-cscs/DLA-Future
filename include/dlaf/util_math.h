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

#include <functional>
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

template <class S, class U,
          std::enable_if_t<std::is_integral<U>::value && std::is_unsigned<U>::value &&
                               std::is_integral<S>::value && std::is_signed<S>::value,
                           int> = 0>
S to_signed(const U unsigned_value) {
  assert(std::numeric_limits<S>::max() > unsigned_value);
  return static_cast<S>(unsigned_value);
}

namespace size_t {
namespace internal {

/// @brief Perform the given binary operation on integer types in size_t arithmetic
///
/// It casts operands to size_t type and then performs the given binary operation @p op
/// @tparam TA and @tparam TB must be an integer types (signed or unsigned)
/// @return @p a @p op @p b
/// @pre @p a >=0 and @p b >= 0
#ifdef DLAF_DOXYGEN
template <typename TA, typename TB, typename BinaryOp>
constexpr std::size_t generic_integer_op(const TA a, const TB b, BinaryOp op);
#else
template <typename TA, typename TB, typename BinaryOp>
constexpr auto generic_integer_op(const TA a, const TB b, BinaryOp op)
    -> std::enable_if_t<std::is_integral<TA>::value && std::is_integral<TB>::value, std::size_t> {
  return op(static_cast<std::size_t>(a), static_cast<std::size_t>(b));
}
#endif

}

/// @brief Perform the sum on integer types in size_t arithmetic
///
/// It casts operands to size_t type and then performs the sum
/// @tparam TA and @tparam TB must be an integer types (signed or unsigned)
/// @return @p a + @p b
/// @pre @p a >=0 and @p b >= 0
template <typename TA, typename TB>
std::size_t sum(const TA a, const TB b) {
  return internal::generic_integer_op(a, b, std::plus<std::size_t>());
}

/// @brief Perform the multiplication on integer types in size_t arithmetic
///
/// It casts operands to size_t type and then performs the multiplication
/// @tparam TA and @tparam TB must be an integer types (signed or unsigned)
/// @return @p a * @p b
/// @pre @p a >=0 and @p b >= 0
template <typename TA, typename TB>
std::size_t mul(const TA a, const TB b) {
  return internal::generic_integer_op(a, b, std::multiplies<std::size_t>());
}

}
}
}
