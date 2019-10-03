//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/util_math.h"

#include "gtest/gtest.h"

using namespace dlaf;
using namespace testing;

TEST(MathUtilTest, CeilDiv) {
  EXPECT_EQ(0, util::ceilDiv(0, 1));
  EXPECT_EQ(0, util::ceilDiv(0, 10));
  EXPECT_EQ(3, util::ceilDiv(3, 1));
  EXPECT_EQ(1, util::ceilDiv(3, 3));
  EXPECT_EQ(2, util::ceilDiv(4, 3));
  EXPECT_EQ(2, util::ceilDiv(5, 3));
  EXPECT_EQ(2, util::ceilDiv(6, 3));

  // signed integers
  EXPECT_EQ(6, util::ceilDiv(36, 7));
  EXPECT_EQ(6l, util::ceilDiv(36l, 7l));
  EXPECT_EQ(6ll, util::ceilDiv(36ll, 7ll));
  EXPECT_TRUE((std::is_same<int, decltype(util::ceilDiv(5, 3))>()));
  EXPECT_TRUE((std::is_same<long, decltype(util::ceilDiv(5l, 3l))>()));
  EXPECT_TRUE((std::is_same<long long, decltype(util::ceilDiv(5ll, 3ll))>()));

  // unsigned integers
  EXPECT_EQ(6u, util::ceilDiv(36u, 7u));
  EXPECT_EQ(6ul, util::ceilDiv(36ul, 7ul));
  EXPECT_EQ(6ull, util::ceilDiv(36ull, 7ull));
  EXPECT_TRUE((std::is_same<unsigned int, decltype(util::ceilDiv(5u, 3u))>()));
  EXPECT_TRUE((std::is_same<unsigned long, decltype(util::ceilDiv(5ul, 3ul))>()));
  EXPECT_TRUE((std::is_same<unsigned long long, decltype(util::ceilDiv(5ull, 3ull))>()));
}

template <typename Type>
class MathUtilTest : public ::testing::Test {};

using IntegerTypes = ::testing::Types<char, short, int, long, long long, unsigned char, unsigned short,
                                      unsigned int, unsigned long, unsigned long long>;

TYPED_TEST_CASE(MathUtilTest, IntegerTypes);

TYPED_TEST(MathUtilTest, CeilDivType) {
  using Type = TypeParam;

  Type num = 36;
  Type den = 7;
  Type res = 6;
  EXPECT_EQ(res, util::ceilDiv(num, den));
  EXPECT_TRUE((std::is_same<Type, decltype(util::ceilDiv(num, den))>()));
}

TYPED_TEST(MathUtilTest, SizeTArithmetic_Sum) {
  using Type = TypeParam;

  constexpr auto type_size = sizeof(Type);
  constexpr auto arithmetic_size = sizeof(std::size_t);

  Type a, b;

  if (type_size < arithmetic_size) {
    a = std::numeric_limits<Type>::max();
    b = std::numeric_limits<Type>::max();
  }
  else {
    a = std::numeric_limits<std::size_t>::max() / 2;
    b = std::numeric_limits<std::size_t>::max() / 2;
  }

  auto expected_result = static_cast<size_t>(a) + static_cast<size_t>(b);

  EXPECT_EQ(expected_result, util::size_t::sum(a, b));
}

TYPED_TEST(MathUtilTest, SizeTArithmetic_Mul) {
  using Type = TypeParam;

  constexpr auto type_size = sizeof(Type);
  constexpr auto arithmetic_size = sizeof(std::size_t);

  Type a, b;

  if (type_size < arithmetic_size) {
    a = std::numeric_limits<Type>::max();
    b = std::numeric_limits<Type>::max();
  }
  else {
    a = std::numeric_limits<std::size_t>::max() / 2;
    b = 2;
  }

  auto expected_result = static_cast<size_t>(a) * static_cast<size_t>(b);

  EXPECT_EQ(expected_result, util::size_t::mul(a, b));
}

TYPED_TEST(MathUtilTest, SizeTArithmetic_SumMul) {
  using Type = TypeParam;

  constexpr auto type_size = sizeof(Type);
  constexpr auto arithmetic_size = sizeof(std::size_t);

  Type b = 2;
  Type a, c;

  if (type_size < arithmetic_size) {
    a = std::numeric_limits<Type>::max() / 2;
    c = std::numeric_limits<Type>::max();
  }
  else {
    a = std::numeric_limits<std::size_t>::max() / 4;
    c = std::numeric_limits<std::size_t>::max() / 2;
  }

  auto expected_result = static_cast<size_t>(a) * static_cast<size_t>(b) + static_cast<size_t>(c);

  using util::size_t::sum;
  using util::size_t::mul;

  EXPECT_EQ(expected_result, sum(mul(a, b), c));
}
