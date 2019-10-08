//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <stdexcept>
#include "gtest/gtest.h"
#include "dlaf/matrix_base.h"

using namespace dlaf;
using namespace testing;

std::vector<SizeType> ms({0, 1, 13, 32, 128});
std::vector<SizeType> ns({0, 1, 16, 32, 128});
std::vector<SizeType> mbs({13, 128});
std::vector<SizeType> nbs({16, 64});

TEST(MatrixBaseTest, DefaultConstructor) {
  MatrixBase obj;

  EXPECT_EQ(GlobalElementSize(0, 0), obj.size());
  EXPECT_EQ(LocalTileSize(0, 0), obj.nrTiles());
  EXPECT_EQ(TileElementSize(1, 1), obj.blockSize());
}

TEST(MatrixBaseTest, Constructor) {
  for (const auto m : ms)
    for (const auto n : ns)
      for (const auto mb : mbs)
        for (const auto nb : nbs) {
          MatrixBase obj({m, n}, {mb, nb});

          EXPECT_EQ(GlobalElementSize(m, n), obj.size());
          EXPECT_EQ(LocalTileSize(util::ceilDiv(m, mb), util::ceilDiv(n, nb)), obj.nrTiles());
          EXPECT_EQ(TileElementSize(mb, nb), obj.blockSize());
        }
}

TEST(MatrixBaseTest, ConstructorExceptions) {
  for (const auto m : ms)
    for (const auto n : ns)
      for (const auto mb : mbs)
        for (const auto nb : nbs) {
          EXPECT_THROW(MatrixBase obj({-1, n}, {mb, nb}), std::invalid_argument);
          EXPECT_THROW(MatrixBase obj({m, -1}, {mb, nb}), std::invalid_argument);
          EXPECT_THROW(MatrixBase obj({m, n}, {0, nb}), std::invalid_argument);
          EXPECT_THROW(MatrixBase obj({m, n}, {mb, 0}), std::invalid_argument);
          EXPECT_THROW(MatrixBase obj({m, n}, {-1, nb}), std::invalid_argument);
          EXPECT_THROW(MatrixBase obj({m, n}, {mb, -1}), std::invalid_argument);
        }
}

TEST(MatrixBaseTest, EqualityOperator) {
  for (const auto m : ms)
    for (const auto n : ns)
      for (const auto mb : mbs)
        for (const auto nb : nbs) {
          MatrixBase obj({m, n}, {mb, nb});
          MatrixBase obj_eq({m, n}, {mb, nb});

          EXPECT_TRUE(obj == obj_eq);
          EXPECT_FALSE(obj != obj_eq);

          std::vector<MatrixBase> objs_ne;
          objs_ne.emplace_back(GlobalElementSize(m + 1, n), TileElementSize(mb, nb));
          objs_ne.emplace_back(GlobalElementSize(m, n + 1), TileElementSize(mb, nb));
          objs_ne.emplace_back(GlobalElementSize(m, n), TileElementSize(mb + 1, nb));
          objs_ne.emplace_back(GlobalElementSize(m, n), TileElementSize(mb, nb + 1));

          for (const auto& obj_ne : objs_ne) {
            EXPECT_TRUE(obj != obj_ne);
            EXPECT_FALSE(obj == obj_ne);
          }
        }
}

TEST(MatrixBaseTest, CopyConstructor) {
  for (const auto m : ms)
    for (const auto n : ns)
      for (const auto mb : mbs)
        for (const auto nb : nbs) {
          MatrixBase obj0({m, n}, {mb, nb});
          MatrixBase obj({m, n}, {mb, nb});
          EXPECT_EQ(obj0, obj);

          MatrixBase obj_copy = obj;
          EXPECT_EQ(obj0, obj);
          EXPECT_EQ(obj, obj_copy);
        }
}

TEST(MatrixBaseTest, MoveConstructor) {
  for (const auto m : ms)
    for (const auto n : ns)
      for (const auto mb : mbs)
        for (const auto nb : nbs) {
          MatrixBase obj0({m, n}, {mb, nb});
          MatrixBase obj({m, n}, {mb, nb});
          EXPECT_EQ(obj0, obj);

          MatrixBase obj_move = std::move(obj);
          EXPECT_EQ(MatrixBase(), obj);
          EXPECT_EQ(obj0, obj_move);
        }
}

TEST(MatrixBaseTest, CopyAssignment) {
  for (const auto m : ms)
    for (const auto n : ns)
      for (const auto mb : mbs)
        for (const auto nb : nbs) {
          MatrixBase obj0({m, n}, {mb, nb});
          MatrixBase obj({m, n}, {mb, nb});
          EXPECT_EQ(obj0, obj);

          MatrixBase obj_copy;
          obj_copy = obj;
          EXPECT_EQ(obj0, obj);
          EXPECT_EQ(obj, obj_copy);
        }
}

TEST(MatrixBaseTest, MoveAssignment) {
  for (const auto m : ms)
    for (const auto n : ns)
      for (const auto mb : mbs)
        for (const auto nb : nbs) {
          MatrixBase obj0({m, n}, {mb, nb});
          MatrixBase obj({m, n}, {mb, nb});
          EXPECT_EQ(obj0, obj);

          MatrixBase obj_move;
          obj_move = std::move(obj);
          EXPECT_EQ(MatrixBase(), obj);
          EXPECT_EQ(obj0, obj_move);
        }
}
