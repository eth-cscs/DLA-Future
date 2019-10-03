//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/common/index2d.h"

#include <array>

#include <gtest/gtest.h>

template <typename IndexType>
using Size2D = dlaf::common::Size2D<IndexType, struct TAG_TEST>;

template <typename IndexType>
class Size2DTest : public ::testing::Test {};

using IndexTypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t>;
TYPED_TEST_CASE(Size2DTest, IndexTypes);

TYPED_TEST(Size2DTest, ConstructorFromParams) {
  TypeParam row = 5;
  TypeParam col = 3;
  Size2D<TypeParam> size(row, col);

  EXPECT_EQ(row, size.rows());
  EXPECT_EQ(col, size.cols());

  EXPECT_TRUE(size.isValid());
}

TYPED_TEST(Size2DTest, ConstructorFromArray) {
  std::array<TypeParam, 2> coords{5, 3};
  Size2D<TypeParam> size(coords);

  EXPECT_EQ(coords[0], size.rows());
  EXPECT_EQ(coords[1], size.cols());

  EXPECT_TRUE(size.isValid());
}

TYPED_TEST(Size2DTest, Comparison) {
  TypeParam row = 5;
  TypeParam col = 3;
  Size2D<TypeParam> size1(row, col);

  std::array<TypeParam, 2> coords{row, col};
  Size2D<TypeParam> size2(coords);

  EXPECT_TRUE(size1 == size2);
  EXPECT_FALSE(size1 != size2);

  Size2D<TypeParam> size3(row + 1, col);
  Size2D<TypeParam> size4(row, col - 1);
  Size2D<TypeParam> size5(row + 4, col - 2);

  EXPECT_TRUE(size1 != size3);
  EXPECT_TRUE(size1 != size4);
  EXPECT_TRUE(size1 != size5);
  EXPECT_FALSE(size1 == size3);
  EXPECT_FALSE(size1 == size4);
  EXPECT_FALSE(size1 == size5);
}
