//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/eigensolver/tridiag_solver.h"

#include "gtest/gtest.h"
//#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

template <typename Type>
class TridiagEigensolverTest : public ::testing::Test {};

using namespace dlaf;
using namespace dlaf::test;

TYPED_TEST_SUITE(TridiagEigensolverTest, RealMatrixElementTypes);

TEST(MatrixIndexPairsGeneration, IndexPairsGeneration) {
  SizeType n = 10;
  auto actual_indices = dlaf::eigensolver::internal::generateSubproblemIndices(n);
  // i_begin, i_middle, i_end
  std::vector<std::tuple<SizeType, SizeType, SizeType>> expected_indices{{0, 0, 1}, {0, 1, 2},
                                                                         {3, 3, 4}, {0, 2, 4},
                                                                         {5, 5, 6}, {5, 6, 7},
                                                                         {8, 8, 9}, {5, 7, 9},
                                                                         {0, 4, 9}};
  ASSERT_TRUE(actual_indices == expected_indices);
}

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

// Initializes square input and output matices of size (n x n) and block size (nb x nb). The matrices are
// discribed by a distribution and an array of input and output tiles respectively. Note that in contrast
// to `Matrix<>`, the tiles are readily available, they are not stored in futures.
//
// The elements of the input matrix are as follows:
//
// `in_ij = i + j`
//
// The output matrix is set to zero.
//
template <class T>
std::tuple<matrix::Distribution, std::vector<matrix::Tile<T, Device::CPU>>,
           std::vector<matrix::Tile<T, Device::CPU>>>
setupMaticesForPermutations(SizeType n, SizeType nb) {
  using dlaf::matrix::test::createTile;
  using dlaf::matrix::test::set;
  matrix::Distribution distr(LocalElementSize(n, n), TileElementSize(nb, nb));
  GlobalTileSize ntiles = distr.nrTiles();

  std::vector<matrix::Tile<T, Device::CPU>> in_tiles;
  std::vector<matrix::Tile<T, Device::CPU>> out_tiles;
  in_tiles.reserve(to_sizet(ntiles.linear_size()));
  out_tiles.reserve(to_sizet(ntiles.linear_size()));

  // Allocate tiles
  for (SizeType j_tile = 0; j_tile < ntiles.cols(); ++j_tile) {
    for (SizeType i_tile = 0; i_tile < ntiles.rows(); ++i_tile) {
      GlobalTileIndex tile_idx(i_tile, j_tile);
      TileElementSize tile_sz = distr.tileSize(tile_idx);
      auto in_tile_fn = [&distr, tile_idx](const TileElementIndex& el_idx) {
        GlobalElementIndex el_idx_gl = distr.globalElementIndex(tile_idx, el_idx);
        return el_idx_gl.row() + el_idx_gl.col();
      };
      in_tiles.push_back(createTile<T, Device::CPU>(std::move(in_tile_fn), tile_sz, tile_sz.rows()));
      out_tiles.push_back(createTile<T, Device::CPU>(tile_sz, tile_sz.rows()));
      set(out_tiles.back(), T(0));
    }
  }

  return std::make_tuple(std::move(distr), std::move(in_tiles), std::move(out_tiles));
}

// The following column permutations are tested:
//
//  ┌─────────────────┐   ┌─────────────────┐
//  │in               │   │out              │
//  │     ┌─┬─┐ ┌─┐ ┌─┤   │                 │
//  │     │ │ │ │ │ │ │   │                 │
//  │     │a│b│ │c│ │d│   │       ┌─┬─┬─┬─┐ │
//  │     │ │ │ │ │ │ │   │       │ │ │ │ │ │
//  │     └─┴─┘ └─┘ └─┤   │       │a│c│b│d│ │
//  │                 │   │       │ │ │ │ │ │
//  │                 │   │       └─┴─┴─┴─┘ │
//  │                 │   │                 │
//  └─────────────────┘   └─────────────────┘
//
TYPED_TEST(TridiagEigensolverTest, ApplyColumnPermutations) {
  using T = TypeParam;
  using dlaf::matrix::test::createTile;
  using dlaf::matrix::test::set;

  SizeType n = 10;
  SizeType nb = 3;
  auto [distr, in_tiles, out_tiles] = setupMaticesForPermutations<T>(n, nb);

  GlobalElementIndex begin(4, 4);
  GlobalElementSize sz(3, 4);
  SizeType in_offset = 2;
  std::vector<SizeType> perm_arr{3, 6, 4, 9};
  dlaf::eigensolver::internal::applyPermutations<TypeParam, Coord::Col>(begin, sz, in_offset, distr,
                                                                        perm_arr.data(), in_tiles,
                                                                        out_tiles);

  // Expected entries in `out_tiles`
  //
  // (tile index in `out_tiles`, index of element in tile, value of element)
  // clang-format off
  std::vector<std::tuple<std::size_t, TileElementIndex, T>> expected_entries {
    // column 4
    {5, TileElementIndex(1, 1), 5},
    {5, TileElementIndex(2, 1), 6},
    {6, TileElementIndex(0, 1), 7},

    // column 5
    {5, TileElementIndex(1, 2), 8},
    {5, TileElementIndex(2, 2), 9},
    {6, TileElementIndex(0, 2), 10},

    // column 6
    {9,  TileElementIndex(1, 0), 6},
    {9,  TileElementIndex(2, 0), 7},
    {10, TileElementIndex(0, 0), 8},

    // column 7
    {9,  TileElementIndex(1, 1), 11},
    {9,  TileElementIndex(2, 1), 12},
    {10, TileElementIndex(0, 1), 13},
  };
  // clang-format on

  for (auto [i_tile, i_el, val] : expected_entries) {
    EXPECT_NEAR(out_tiles[i_tile](i_el), val, 1e-7);
  }
}

// The following row permutations are tested:
//
// ┌──────────────────────┐    ┌──────────────────────┐
// │in                    │    │ out                  │
// │     ****** a         │    │                      │
// │                      │    │                      │
// │                      │    │                      │
// │     ****** b         │    │                      │
// │                      │    │         c ******     │
// │                      │    │         b ******     │
// │     ****** c         │    │         a ******     │
// │                      │    │                      │
// │                      │    │                      │
// └──────────────────────┘    └──────────────────────┘
// where the (i, j) element of `in` is `i + j`
//
TYPED_TEST(TridiagEigensolverTest, ApplyRowPermutations) {
  using T = TypeParam;
  using dlaf::matrix::test::createTile;
  using dlaf::matrix::test::set;

  SizeType n = 10;
  SizeType nb = 3;
  auto [distr, in_tiles, out_tiles] = setupMaticesForPermutations<T>(n, nb);

  GlobalElementIndex begin(7, 3);
  GlobalElementSize sz(3, 6);
  SizeType in_offset = 2;
  std::vector<SizeType> perm_arr{8, 4, 1};
  dlaf::eigensolver::internal::applyPermutations<TypeParam, Coord::Row>(begin, sz, in_offset, distr,
                                                                        perm_arr.data(), in_tiles,
                                                                        out_tiles);

  // Expected entries in `out_tiles`
  //
  // (tile index in `out_tiles`, index of element in tile, value of element)
  // clang-format off
  std::vector<std::tuple<std::size_t, TileElementIndex, T>> expected_entries {
    // row index 7
    {6,  TileElementIndex(1, 0), 10},
    {6,  TileElementIndex(1, 1), 11},
    {6,  TileElementIndex(1, 2), 12},
    {10, TileElementIndex(1, 0), 13},
    {10, TileElementIndex(1, 1), 14},
    {10, TileElementIndex(1, 2), 15},

    // row index 8
    {6,  TileElementIndex(2, 0), 6 },
    {6,  TileElementIndex(2, 1), 7 },
    {6,  TileElementIndex(2, 2), 8 },
    {10, TileElementIndex(2, 0), 9 },
    {10, TileElementIndex(2, 1), 10},
    {10, TileElementIndex(2, 2), 11},

    // row index 9
    {7,  TileElementIndex(0, 0), 3},
    {7,  TileElementIndex(0, 1), 4},
    {7,  TileElementIndex(0, 2), 5},
    {11, TileElementIndex(0, 0), 6},
    {11, TileElementIndex(0, 1), 7},
    {11, TileElementIndex(0, 2), 8},

  };
  // clang-format on

  for (auto [i_tile, i_el, val] : expected_entries) {
    EXPECT_NEAR(out_tiles[i_tile](i_el), val, 1e-7);
  }
}

// TYPED_TEST(TridiagEigensolverTest, AssembleZVector) {
//   SizeType n = 10;
//   SizeType nb = 2;
//
//   SizeType i_begin = 2;
//   SizeType i_middle = 4;
//   SizeType i_end = 6;
//   matrix::Matrix<TypeParam, Device::CPU> mat_ev(LocalElementSize(n, n), TileElementSize(nb, nb));
//   matrix::Matrix<TypeParam, Device::CPU> z(LocalElementSize(n, 1), TileElementSize(nb, 1));
//
//   eigensolver::internal::assembleZVec(i_begin, i_middle, i_end, mat_ev, z);
// }

// TYPED_TEST(TridiagEigensolverTest, AssembleDiag) {
//   SizeType n = 10;
//   SizeType nb = 2;
//
//   SizeType i_begin = 0;
//   SizeType i_end = 4;
//   matrix::Matrix<TypeParam, Device::CPU> mat_a(LocalElementSize(n, n), TileElementSize(nb, nb));
//   matrix::Matrix<TypeParam, Device::CPU> d(LocalElementSize(n, 1), TileElementSize(nb, 1));
//
//   eigensolver::internal::assembleDiag(i_begin, i_end, mat_a, d);
// }

TYPED_TEST(TridiagEigensolverTest, CuppensDecomposition) {
  using matrix::test::createTile;

  SizeType sz = 10;
  auto laplace1d_fn = [](const TileElementIndex& idx) {
    if (idx.col() == 0)
      return TypeParam(2);
    else
      return TypeParam(-1);
  };

  TileElementSize tile_size(sz, 2);
  auto top = createTile<TypeParam, Device::CPU>(laplace1d_fn, tile_size, sz);
  auto bottom = createTile<TypeParam, Device::CPU>(laplace1d_fn, tile_size, sz);

  eigensolver::internal::cuppensTileDecomposition(top, bottom);

  auto expected_top = createTile<TypeParam, Device::CPU>(laplace1d_fn, tile_size, sz);
  auto expected_bottom = createTile<TypeParam, Device::CPU>(laplace1d_fn, tile_size, sz);
  expected_top(TileElementIndex(sz - 1, 0)) = TypeParam(3);
  expected_bottom(TileElementIndex(0, 0)) = TypeParam(3);

  CHECK_TILE_NEAR(expected_top, top, TypeUtilities<TypeParam>::error, TypeUtilities<TypeParam>::error);
  CHECK_TILE_NEAR(expected_bottom, bottom, TypeUtilities<TypeParam>::error,
                  TypeUtilities<TypeParam>::error);
}

TYPED_TEST(TridiagEigensolverTest, CorrectnessLocal) {
  using RealParam = BaseType<TypeParam>;

  SizeType n = 10;
  SizeType nb = 2;

  matrix::Matrix<RealParam, Device::CPU> mat_a(LocalElementSize(n, 2), TileElementSize(nb, 2));
  matrix::Matrix<TypeParam, Device::CPU> mat_ev(LocalElementSize(n, n), TileElementSize(nb, nb));

  // Tridiagonal matrix : 1D Laplacian
  auto mat_a_fn = [](GlobalElementIndex el) {
    if (el.col() == 0)
      // diagonal
      return RealParam(-1);
    else
      // off-diagoanl
      return RealParam(2);
  };
  matrix::util::set(mat_a, std::move(mat_a_fn));

  // eigensolver::tridiagSolver<Backend::MC>(mat_a, mat_ev);

  // TODO: checks
}
