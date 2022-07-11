//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cstddef>
#include <functional>
#include <type_traits>
#include <vector>

#include "dlaf/types.h"

/// @file

namespace dlaf::util {

// Interleaves two intervals of length `l` split in blocks of size `b` starting at offsets `o1` and
// `o2` respectively and returns an array of indices where the splits have occured.
//
// o1
//  │
//  └► │   │   │   │   │
//    ─┴───┴───┴───┴───┴───  ◄─┐
//               ▲             │
// o2      b ─┬──┘              l
// │          │                │
// └─►  │   │ ▼ │   │   │      │
//    ──┴───┴───┴───┴───┴──  ◄─┘
//
inline std::vector<SizeType> interleaveSplits(SizeType l, SizeType b, SizeType o1, SizeType o2) {
  DLAF_ASSERT(l > 0, l);
  DLAF_ASSERT(b > 0, b);
  DLAF_ASSERT(o1 >= 0 && o1 <= b, o1, b);
  DLAF_ASSERT(o2 >= 0 && o2 <= b, o2, b);

  // Set small and big from offsets o1 and o2 s.t small <= big
  SizeType small = o1;
  SizeType big = o2;
  if (small > big)
    std::swap(small, big);

  // Reserve enough memory for array of splits
  std::vector<SizeType> splits;
  splits.reserve(2 * to_sizet(l / b) + 2);

  splits.push_back(0);
  for (SizeType i = small, j = big; i < l || j < l; i += b, j += b) {
    if (splits.back() != i && i < l)
      splits.push_back(i);
    if (splits.back() != j && j < l)
      splits.push_back(j);
  }
  splits.push_back(l);
  return splits;
}

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
    -> std::enable_if_t<std::is_integral_v<IntType>, IntType> {
  return (num + den - 1) / den;
}
#endif

/// Returns true if @p a and @p b are both positive or both negative
template <class T>
bool sameSign(T a, T b) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "T must be a real number.");
  return std::signbit(a) == std::signbit(b);
}

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
SizeType sum(const TA a, const TB b) {
  return dlaf::util::internal::generic_integer_op<SizeType>(a, b, std::plus<SizeType>());
}

/// Perform the multiplication on integer types in ptrdiff_t arithmetic.
///
/// It casts operands to std::ptrdiff_t type and then performs the multiplication.
/// @tparam TA and @tparam TB must be integer types (signed or unsigned),
/// @return @p a * @p b.
/// @pre @p a and @p b can be stored in std::ptrdiff_t,
/// @pre it must be possible to store the result in std::ptrdiff_t.
template <typename TA, typename TB>
SizeType mul(const TA a, const TB b) {
  return dlaf::util::internal::generic_integer_op<SizeType>(a, b, std::multiplies<SizeType>());
}

}
}
