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

TEST(Index2D, ConstructorDefault) {
  Index2D index;

  EXPECT_LT(index.row(), 0);
  EXPECT_LT(index.col(), 0);

  EXPECT_FALSE(index.isValid());
}

TEST(Index2D, ConstructorFromParams) {
  Index2D index(5, 3);

  EXPECT_EQ(5, index.row());
  EXPECT_EQ(3, index.col());

  EXPECT_TRUE(index.isValid());
}

TEST(Index2D, ConstructorFromArray) {
  Index2D index(std::array<int, 2>{5, 3});

  EXPECT_EQ(5, index.row());
  EXPECT_EQ(3, index.col());

  EXPECT_TRUE(index.isValid());
}

TEST(Index2D, BoundaryCheck) {
  Index2D boundary(3, 6);

  EXPECT_TRUE(boundary.isValid());

  auto index_tests = std::vector<std::pair<Index2D, bool>>{
    std::make_pair(Index2D(0, 4), true),    // in bounds
    std::make_pair(Index2D(3, 6), false),   // out (on edge)
    std::make_pair(Index2D(5, 9), false),   // out of bounds
    std::make_pair(Index2D(3, 5), false),   // out of row
    std::make_pair(Index2D(2, 6), false),   // out of col
  };

  for (auto & test : index_tests) {
    EXPECT_TRUE(test.first.isValid());
    EXPECT_EQ(test.first < boundary, test.second);
  }
}
