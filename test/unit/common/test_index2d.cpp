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
#include <sstream>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

template <typename IndexType>
using Index2D = dlaf::common::Index2D<IndexType, struct TAG_TEST>;

template <typename IndexType>
using Size2D = dlaf::common::Size2D<IndexType, struct TAG_TEST>;

template <typename IndexType>
class Index2DTest : public ::testing::Test {};

using IndexTypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t>;
TYPED_TEST_SUITE(Index2DTest, IndexTypes);

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
  auto index_tests =
      std::vector<std::pair<TypeParam, TypeParam>>{std::make_pair(-10, -10), std::make_pair(-1, -1),
                                                   std::make_pair(-15, 3), std::make_pair(8, -7)};

  for (auto& test : index_tests) {
    EXPECT_THROW(Index2D<TypeParam>(test.first, test.second), std::invalid_argument);
    EXPECT_THROW(Index2D<TypeParam>(std::array<TypeParam, 2>{test.first, test.second}),
                 std::invalid_argument);
  }
}

TYPED_TEST(Index2DTest, BoundaryCheck) {
  dlaf::common::Size2D<TypeParam, struct TAG_TEST> boundary(3, 6);

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
    EXPECT_EQ(test.second, test.first.isIn(boundary));
  }
}

TYPED_TEST(Index2DTest, Comparison) {
  TypeParam row = 5;
  TypeParam col = 3;
  Index2D<TypeParam> index1(row, col);

  std::array<TypeParam, 2> coords{row, col};
  Index2D<TypeParam> index2(coords);

  EXPECT_TRUE(index1 == index2);
  EXPECT_FALSE(index1 != index2);

  Index2D<TypeParam> index3(row + 1, col);
  Index2D<TypeParam> index4(row, col - 1);
  Index2D<TypeParam> index5(row + 4, col - 2);

  EXPECT_TRUE(index1 != index3);
  EXPECT_TRUE(index1 != index4);
  EXPECT_TRUE(index1 != index5);
  EXPECT_FALSE(index1 == index3);
  EXPECT_FALSE(index1 == index4);
  EXPECT_FALSE(index1 == index5);
}

TYPED_TEST(Index2DTest, Transpose) {
  Index2D<TypeParam> index1(7, 13);
  std::array<TypeParam, 2> coords{9, 6};
  Index2D<TypeParam> index2(coords);

  index1.transpose();
  EXPECT_EQ(Index2D<TypeParam>(13, 7), index1);
  index1.transpose();
  EXPECT_EQ(Index2D<TypeParam>(7, 13), index1);

  index2.transpose();
  EXPECT_EQ(Index2D<TypeParam>(6, 9), index2);
}

TYPED_TEST(Index2DTest, Print) {
  Index2D<TypeParam> index1(7, 13);
  std::array<TypeParam, 2> coords{9, 6};
  Index2D<TypeParam> index2(coords);

  std::stringstream s;
  s << index1;
  EXPECT_EQ("(7, 13)", s.str());

  s.str("");
  s << index2;
  EXPECT_EQ("(9, 6)", s.str());
}

TYPED_TEST(Index2DTest, ComputeLinearIndex) {
  using dlaf::common::Ordering;

#ifdef DLAF_ASSERT_MODERATE_ENABLE
  {
    const Index2D<TypeParam> index(13, 26);
    std::vector<Size2D<TypeParam>> configs{{
        Size2D<TypeParam>{13, 26},  // out-of-rows & out-of-cols
        Size2D<TypeParam>{13, 39},  // out-of-rows
        Size2D<TypeParam>{26, 26},  // out-of-cols
    }};

    std::array<Ordering, 2> orderings{Ordering::RowMajor, Ordering::ColumnMajor};

    for (const auto& ordering : orderings) {
      for (const Size2D<TypeParam>& size : configs) {
        EXPECT_DEATH(computeLinearIndex(ordering, index, {size.rows(), size.cols()}), "[ERROR]");
        EXPECT_DEATH(computeLinearIndex(ordering, index, Size2D<TypeParam>(size)), "[ERROR]");
      }
    }
  }
#endif

  {
    const Index2D<TypeParam> index{3, 2};
    EXPECT_EQ(13, computeLinearIndex(Ordering::ColumnMajor, index, {5, 26}));
    EXPECT_EQ(13, computeLinearIndex(Ordering::ColumnMajor, index, Size2D<TypeParam>{5, 26}));
  }

  {
    const Index2D<TypeParam> index{2, 3};
    EXPECT_EQ(13, computeLinearIndex(Ordering::RowMajor, index, {26, 5}));
    EXPECT_EQ(13, computeLinearIndex(Ordering::RowMajor, index, Size2D<TypeParam>{26, 5}));
  }
}
