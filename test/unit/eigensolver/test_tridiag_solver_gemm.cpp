//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/eigensolver/tridiag_solver/gemm.h"

#include "gtest/gtest.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::test;
using namespace dlaf::eigensolver::internal;

template <typename Type>
class TridiagEigensolverGEMMTest : public ::testing::Test {};
TYPED_TEST_SUITE(TridiagEigensolverGEMMTest, RealMatrixElementTypes);

TEST(Splits, DifferentOffsets) {
  SizeType l = 10;
  SizeType b = 3;
  SizeType o1 = 1;
  SizeType o2 = 2;
  std::vector<SizeType> splits = dlaf::eigensolver::internal::interleaveSplits(l, b, o1, o2);

  std::vector<SizeType> expected_splits{0, 1, 2, 4, 5, 7, 8, 10};
  ASSERT_TRUE(splits == expected_splits);
}

TEST(Splits, EqualOffsets) {
  SizeType l = 10;
  SizeType b = 3;
  SizeType o1 = 2;
  SizeType o2 = 2;
  std::vector<SizeType> splits = dlaf::eigensolver::internal::interleaveSplits(l, b, o1, o2);

  std::vector<SizeType> expected_splits{0, 2, 5, 8, 10};
  ASSERT_TRUE(splits == expected_splits);
}

TEST(Splits, SingleZeroOffset) {
  SizeType l = 10;
  SizeType b = 3;
  SizeType o1 = 2;
  SizeType o2 = 0;
  std::vector<SizeType> splits = dlaf::eigensolver::internal::interleaveSplits(l, b, o1, o2);

  std::vector<SizeType> expected_splits{0, 2, 3, 5, 6, 8, 9, 10};
  ASSERT_TRUE(splits == expected_splits);
}

TEST(Splits, ZeroOffsets) {
  SizeType l = 10;
  SizeType b = 3;
  SizeType o1 = 0;
  SizeType o2 = 0;
  std::vector<SizeType> splits = dlaf::eigensolver::internal::interleaveSplits(l, b, o1, o2);

  std::vector<SizeType> expected_splits{0, 3, 6, 9, 10};
  ASSERT_TRUE(splits == expected_splits);
}

TYPED_TEST(TridiagEigensolverGEMMTest, SubmatrixGEMM) {
  using dlaf::matrix::test::createTile;
  using dlaf::matrix::test::set;
  using T = TypeParam;

  SizeType n = 10;
  SizeType nb = 3;

  matrix::Distribution distr(LocalElementSize(n, n), TileElementSize(nb, nb));
  GlobalTileSize ntiles = distr.nrTiles();

  std::vector<matrix::Tile<T, Device::CPU>> tiles_a;
  std::vector<matrix::Tile<T, Device::CPU>> tiles_b;
  std::vector<matrix::Tile<T, Device::CPU>> tiles_c;
  tiles_a.reserve(to_sizet(ntiles.linear_size()));
  tiles_b.reserve(to_sizet(ntiles.linear_size()));
  tiles_c.reserve(to_sizet(ntiles.linear_size()));

  // Allocate tiles
  for (SizeType j_tile = 0; j_tile < ntiles.cols(); ++j_tile) {
    for (SizeType i_tile = 0; i_tile < ntiles.rows(); ++i_tile) {
      GlobalTileIndex tile_idx(i_tile, j_tile);
      TileElementSize tile_sz = distr.tileSize(tile_idx);
      tiles_a.push_back(createTile<T, Device::CPU>(tile_sz, tile_sz.rows()));
      tiles_b.push_back(createTile<T, Device::CPU>(tile_sz, tile_sz.rows()));
      tiles_c.push_back(createTile<T, Device::CPU>(tile_sz, tile_sz.rows()));
      set(tiles_a.back(), T(1));
      set(tiles_b.back(), T(2));
      set(tiles_c.back(), T(0));
    }
  }

  GlobalElementIndex idx_a(1, 2);
  GlobalElementIndex idx_b(3, 3);
  GlobalElementIndex idx_c(3, 6);

  SizeType len_m = 7;
  SizeType len_n = 4;
  SizeType len_k = 6;

  submatrixGEMM(len_m, len_n, len_k, idx_a, idx_b, idx_c, distr, tiles_a, tiles_b, tiles_c);

  // Expected entries in `tiles_c`
  //
  // (tile index in `tiles_c`, index of element in tile, value of element)
  // clang-format off
  std::vector<std::tuple<std::size_t, TileElementIndex, T>> expected_entries {
    // tile (1, 2)
    {9, TileElementIndex(0, 0), 12},
    {9, TileElementIndex(1, 0), 12},
    {9, TileElementIndex(2, 0), 12},
    {9, TileElementIndex(0, 1), 12},
    {9, TileElementIndex(1, 1), 12},
    {9, TileElementIndex(2, 1), 12},
    {9, TileElementIndex(0, 2), 12},
    {9, TileElementIndex(1, 2), 12},
    {9, TileElementIndex(2, 2), 12},

    // tile (2, 2)
    {10, TileElementIndex(0, 0), 12},
    {10, TileElementIndex(1, 0), 12},
    {10, TileElementIndex(2, 0), 12},
    {10, TileElementIndex(0, 1), 12},
    {10, TileElementIndex(1, 1), 12},
    {10, TileElementIndex(2, 1), 12},
    {10, TileElementIndex(0, 2), 12},
    {10, TileElementIndex(1, 2), 12},
    {10, TileElementIndex(2, 2), 12},

    // tile (3, 2)
    {11, TileElementIndex(0, 0), 12},
    {11, TileElementIndex(0, 1), 12},
    {11, TileElementIndex(0, 2), 12},

    // tile (1, 3)
    {13, TileElementIndex(0, 0), 12},
    {13, TileElementIndex(1, 0), 12},
    {13, TileElementIndex(2, 0), 12},

    // tile (2, 3)
    {14, TileElementIndex(0, 0), 12},
    {14, TileElementIndex(1, 0), 12},
    {14, TileElementIndex(2, 0), 12},

    // tile (3, 3)
    {15, TileElementIndex(0, 0), 12},
  };
  // clang-format on

  for (auto [i_tile, i_el, val] : expected_entries) {
    EXPECT_NEAR(tiles_c[i_tile](i_el), val, 1e-7);
  }
}
