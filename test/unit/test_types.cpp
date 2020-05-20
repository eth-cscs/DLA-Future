//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/types.h"

#include <limits>

#include <gtest/gtest.h>

#include "dlaf/common/utils.h"

using dlaf::to_signed;
using dlaf::to_unsigned;
using dlaf::integral_cast;
using dlaf::common::internal::source_location;

const char* ERROR_MESSAGE = "\\[ERROR\\]";

template <class T>
const auto LOWER_BOUND = std::numeric_limits<T>::min;

template <class T>
const auto UPPER_BOUND = std::numeric_limits<T>::max;

/// Check that no alteration happens during cast From -> To with @param cast_func and dlaf::integral_cast
///
/// Given template type parameters @tparam From and @tparam To,
/// This function calls @p cast_func with a @tparam From parameter, then
/// it compares the return value of @p cast_func and dlaf::integral_cast, of type To, with the
/// static_cast<To> of the value in From
template <class From, class To>
void TEST_CAST(To (*cast_func)(From), const From value, const source_location origin) {
  ::testing::ScopedTrace trace(origin.filename, static_cast<int>(origin.line), "");
  EXPECT_EQ(static_cast<To>(value), cast_func(value));
  EXPECT_EQ(static_cast<To>(value), (integral_cast<To, From>(value)));
}

/// Check that an error is generated during casting operation From -> To
template <class From, class To>
void TEST_CAST_FAIL(To (*cast_func)(From), const From value, const source_location origin) {
  ::testing::ScopedTrace trace(origin.filename, static_cast<int>(origin.line), "");
  EXPECT_DEATH(cast_func(value), ERROR_MESSAGE);
  EXPECT_DEATH((integral_cast<To, From>(value)), ERROR_MESSAGE);
}

#define DLAF_TEST_CAST(From, To, ...) TEST_CAST<From, To>(__VA_ARGS__, SOURCE_LOCATION())

#ifdef DLAF_ASSERT_MODERATE_ENABLE
#define DLAF_TEST_CAST_FAIL(From, To, ...) TEST_CAST_FAIL<From, To>(__VA_ARGS__, SOURCE_LOCATION())
#else
#define DLAF_TEST_CAST_FAIL(From, To, ...)
#endif

TEST(ToSigned, FromUnsigned) {
  using To = int16_t;

  // UP CAST
  //       |---|
  // |-----|-----|
  DLAF_TEST_CAST(uint8_t, To, to_signed, 13);
  DLAF_TEST_CAST(uint8_t, To, to_signed, LOWER_BOUND<uint8_t>());
  DLAF_TEST_CAST(uint8_t, To, to_signed, UPPER_BOUND<uint8_t>());

  // SAME CAST
  //       |-----------|
  // |-----|-----|
  DLAF_TEST_CAST(uint16_t, To, to_signed, 13);
  DLAF_TEST_CAST(uint16_t, To, to_signed, LOWER_BOUND<uint16_t>());
  DLAF_TEST_CAST(uint16_t, To, to_signed, UPPER_BOUND<To>());
  DLAF_TEST_CAST_FAIL(uint16_t, To, to_signed, UPPER_BOUND<uint16_t>());

  // DOWN CAST
  //       |-----------|
  // |-----|-----|
  DLAF_TEST_CAST(uint32_t, To, to_signed, 13);
  DLAF_TEST_CAST(uint32_t, To, to_signed, LOWER_BOUND<uint32_t>());
  DLAF_TEST_CAST(uint32_t, To, to_signed, UPPER_BOUND<To>());
  DLAF_TEST_CAST_FAIL(uint32_t, To, to_signed, UPPER_BOUND<uint32_t>());
}

TEST(ToSigned, FromSigned) {
  using To = int16_t;

  // UP CAST
  //   |---|---|
  // |-----|-----|
  DLAF_TEST_CAST(int8_t, To, to_signed, 13);
  DLAF_TEST_CAST(int8_t, To, to_signed, LOWER_BOUND<int8_t>());
  DLAF_TEST_CAST(int8_t, To, to_signed, UPPER_BOUND<int8_t>());

  //  SAME CAST
  // |-----|-----|
  // |-----|-----|
  DLAF_TEST_CAST(int16_t, To, to_signed, 13);
  DLAF_TEST_CAST(int16_t, To, to_signed, LOWER_BOUND<int16_t>());
  DLAF_TEST_CAST(int16_t, To, to_signed, UPPER_BOUND<int16_t>());

  // DOWN CAST
  // |-----|-----|
  //   |---|---|
  DLAF_TEST_CAST(int32_t, To, to_signed, 13);
  DLAF_TEST_CAST(int32_t, To, to_signed, LOWER_BOUND<To>());
  DLAF_TEST_CAST(int32_t, To, to_signed, UPPER_BOUND<To>());
  DLAF_TEST_CAST_FAIL(int32_t, To, to_signed, LOWER_BOUND<int32_t>());
  DLAF_TEST_CAST_FAIL(int32_t, To, to_signed, UPPER_BOUND<int32_t>());
}

TEST(ToUnsigned, FromSigned) {
  using To = uint16_t;

  // UP CAST
  // |-----|-----|
  //       |-----------|
  DLAF_TEST_CAST(int8_t, To, to_unsigned, 13);
  DLAF_TEST_CAST(int8_t, To, to_unsigned, LOWER_BOUND<To>());
  DLAF_TEST_CAST_FAIL(int8_t, To, to_unsigned, LOWER_BOUND<int8_t>());
  DLAF_TEST_CAST(int8_t, To, to_unsigned, UPPER_BOUND<int8_t>());

  // SAME CAST
  // |-----|-----|
  //       |-----------|
  DLAF_TEST_CAST(int16_t, To, to_unsigned, 13);
  DLAF_TEST_CAST(int16_t, To, to_unsigned, LOWER_BOUND<To>());
  DLAF_TEST_CAST_FAIL(int16_t, To, to_unsigned, LOWER_BOUND<int16_t>());
  DLAF_TEST_CAST(int16_t, To, to_unsigned, UPPER_BOUND<int16_t>());

  // DOWN CAST
  // |-----|-----|
  //       |---|
  DLAF_TEST_CAST(int32_t, To, to_unsigned, 13);
  DLAF_TEST_CAST(int32_t, To, to_unsigned, LOWER_BOUND<To>());
  DLAF_TEST_CAST(int32_t, To, to_unsigned, UPPER_BOUND<To>());
  DLAF_TEST_CAST_FAIL(int32_t, To, to_unsigned, LOWER_BOUND<int32_t>());
  DLAF_TEST_CAST_FAIL(int32_t, To, to_unsigned, UPPER_BOUND<int32_t>());
}

TEST(ToUnsigned, FromUnsigned) {
  using To = uint16_t;

  // UP CAST
  // |---|
  // |-----|
  DLAF_TEST_CAST(uint8_t, To, to_unsigned, 13);
  DLAF_TEST_CAST(uint8_t, To, to_unsigned, LOWER_BOUND<uint8_t>());
  DLAF_TEST_CAST(uint8_t, To, to_unsigned, UPPER_BOUND<uint8_t>());

  // SAME CAST
  // |-----|
  // |-----|
  DLAF_TEST_CAST(uint16_t, To, to_unsigned, 13);
  DLAF_TEST_CAST(uint16_t, To, to_unsigned, LOWER_BOUND<uint16_t>());
  DLAF_TEST_CAST(uint16_t, To, to_unsigned, UPPER_BOUND<uint16_t>());

  // DOWN CAST
  // |-----|
  // |---|
  DLAF_TEST_CAST(uint32_t, To, to_unsigned, 13);
  DLAF_TEST_CAST(uint32_t, To, to_unsigned, LOWER_BOUND<uint32_t>());
  DLAF_TEST_CAST(uint32_t, To, to_unsigned, UPPER_BOUND<To>());
  DLAF_TEST_CAST_FAIL(uint32_t, To, to_unsigned, UPPER_BOUND<uint32_t>());
}
