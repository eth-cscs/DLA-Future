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

using dlaf::common::Index2D;

template <typename IndexType>
class Index2DTest : public ::testing::Test {};

using IndexTypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t>;
TYPED_TEST_CASE(Index2DTest, IndexTypes);

TYPED_TEST(Index2DTest, ConstructorDefault) {
  Index2D<TypeParam> index;

  EXPECT_LT(index.row(), 0);
  EXPECT_LT(index.col(), 0);

  EXPECT_FALSE(index.isValid());
}

TYPED_TEST(Index2DTest, ConstructorFromParams) {
  TypeParam row = 5;
  TypeParam col = 3;
  Index2D<TypeParam> index(row, col);

  EXPECT_EQ(row, index.row());
  EXPECT_EQ(col, index.col());

  EXPECT_TRUE(index.isValid());
}

TYPED_TEST(Index2DTest, ConstructorFromArray) {
  std::array<TypeParam, 2> coords{5, 3};
  Index2D<TypeParam> index(coords);

  EXPECT_EQ(coords[0], index.row());
  EXPECT_EQ(coords[1], index.col());

  EXPECT_TRUE(index.isValid());
}

TYPED_TEST(Index2DTest, Negative) {
  auto index_tests = std::vector<std::pair<TypeParam, TypeParam>>{
    std::make_pair(-10, -10), std::make_pair(-1, -1), std::make_pair(-15, 3), std::make_pair(8, -7)
  };

  for (auto& test : index_tests) {
    EXPECT_THROW(Index2D<TypeParam>(test.first, test.second), std::runtime_error);
    EXPECT_THROW(Index2D<TypeParam>(std::array<TypeParam, 2>{test.first, test.second}), std::runtime_error);
  }
}

TYPED_TEST(Index2DTest, BoundaryCheck) {
  Index2D<TypeParam> boundary(3, 6);

  EXPECT_TRUE(boundary.isValid());

  auto index_tests = std::vector<std::pair<Index2D<TypeParam>, bool>>{
      std::make_pair(Index2D<TypeParam>(0, 4), true),   // in bounds
      std::make_pair(Index2D<TypeParam>(3, 6), false),  // out (on edge)
      std::make_pair(Index2D<TypeParam>(5, 9), false),  // out of bounds
      std::make_pair(Index2D<TypeParam>(3, 5), false),  // out of row
      std::make_pair(Index2D<TypeParam>(2, 6), false),  // out of col
  };

  for (auto& test : index_tests) {
    EXPECT_TRUE(test.first.isValid());
    EXPECT_EQ(test.first < boundary, test.second);
  }
}
