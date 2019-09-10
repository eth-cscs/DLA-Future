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

  EXPECT_FALSE(static_cast<bool>(index));

  EXPECT_FALSE(index < Index2D(6, 4));    // both in bounds
  EXPECT_FALSE(index < Index2D(5, 3));    // both out of bounds
  EXPECT_FALSE(index < Index2D(5, 4));    // row out of bounds
  EXPECT_FALSE(index < Index2D(6, 3));    // col out of bounds
}

TEST(Index2D, ConstructorFromParams) {
  Index2D index(5, 3);

  EXPECT_EQ(5, index.row());
  EXPECT_EQ(3, index.col());

  EXPECT_TRUE(static_cast<bool>(index));

  EXPECT_TRUE(index < Index2D(6, 4));     // both in limit

  EXPECT_FALSE(index < Index2D(5, 3));    // both out of bounds

  EXPECT_FALSE(index < Index2D(5, 4));    // row out of bounds
  EXPECT_FALSE(index < Index2D(6, 3));    // col out of bounds
}

TEST(Index2D, ConstructorFromArray) {
  Index2D index(std::array<int, 2>{5, 3});

  EXPECT_EQ(5, index.row());
  EXPECT_EQ(3, index.col());

  EXPECT_TRUE(static_cast<bool>(index));

  EXPECT_TRUE(index < Index2D(6, 4));     // both in limit

  EXPECT_FALSE(index < Index2D(5, 3));    // both out of bounds

  EXPECT_FALSE(index < Index2D(5, 4));    // row out of bounds
  EXPECT_FALSE(index < Index2D(6, 3));    // col out of bounds
}
