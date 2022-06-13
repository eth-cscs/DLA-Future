//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "gtest/gtest.h"

#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/permutations/general.h"
#include "dlaf/permutations/general/impl.h"
#include "dlaf/util_matrix.h"

#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::test;

template <typename Type>
class PermutationsTestCPU : public ::testing::Test {};
TYPED_TEST_SUITE(PermutationsTestCPU, MatrixElementTypes);

#ifdef DLAF_WITH_CUDA
template <typename Type>
class PermutationsTestGPU : public ::testing::Test {};
TYPED_TEST_SUITE(PermutationsTestGPU, MatrixElementTypes);
#endif

// Initializes square input and output matices of size (n x n) and block size (nb x nb). The matrices are
// described by a distribution and an array of input and output tiles respectively. Note that in contrast
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
setupMatricesForPermutations(SizeType n, SizeType nb) {
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
template <Device D, class T>
void testApplyColumnPermutations(SizeType n, SizeType nb) {
  using dlaf::matrix::test::createTile;
  using dlaf::matrix::test::set;

  auto [distr, in_tiles, out_tiles] = setupMatricesForPermutations<T>(n, nb);

  GlobalElementIndex begin(4, 4);
  GlobalElementSize sz(3, 4);
  SizeType in_offset = 2;
  std::vector<SizeType> perm_arr{3, 6, 4, 9};

  dlaf::permutations::internal::applyPermutations<T, Coord::Col>(begin, sz, in_offset, distr,
                                                                 perm_arr.data(), in_tiles, out_tiles);

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
    EXPECT_EQ(out_tiles[i_tile](i_el), val);
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
template <Device D, class T>
void testApplyRowPermutations(SizeType n, SizeType nb) {
  using dlaf::matrix::test::createTile;
  using dlaf::matrix::test::set;

  auto [distr, in_tiles, out_tiles] = setupMatricesForPermutations<T>(n, nb);

  GlobalElementIndex begin(7, 3);
  GlobalElementSize sz(3, 6);
  SizeType in_offset = 2;
  std::vector<SizeType> perm_arr{8, 4, 1};
  dlaf::permutations::internal::applyPermutations<T, Coord::Row>(begin, sz, in_offset, distr,
                                                                 perm_arr.data(), in_tiles, out_tiles);

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
    EXPECT_EQ(out_tiles[i_tile](i_el), val);
  }
}

// Permute columns or rows in reverse order.
// Each column or row of the input matrix is has it's index as a value.
template <Backend B, Device D, class T, Coord C>
void testPermutations(SizeType n, SizeType nb) {
  Matrix<SizeType, Device::CPU> perms(LocalElementSize(n, 1), TileElementSize(nb, 1));
  Matrix<T, Device::CPU> mat_in_h(LocalElementSize(n, n), TileElementSize(nb, nb));
  Matrix<T, Device::CPU> mat_out_h(LocalElementSize(n, n), TileElementSize(nb, nb));

  dlaf::matrix::util::set(perms, [n](GlobalElementIndex i) { return n - 1 - i.row(); });
  dlaf::matrix::util::set(mat_in_h, [](GlobalElementIndex i) { return T(i.get<C>()); });

  {
    matrix::MatrixMirror<T, D, Device::CPU> mat_in(mat_in_h);
    matrix::MatrixMirror<T, D, Device::CPU> mat_out(mat_out_h);

    SizeType i_begin = 0;
    SizeType i_end = perms.distribution().nrTiles().rows() - 1;
    permutations::permute<B, D, T, C>(i_begin, i_end, perms, mat_in.get(), mat_out.get());
  }

  auto expected_out = [n](const GlobalElementIndex i) { return T(n - 1 - i.get<C>()); };
  CHECK_MATRIX_EQ(expected_out, mat_out_h);
}

TYPED_TEST(PermutationsTestCPU, ApplyColumnPermutations) {
  SizeType n = 10;
  SizeType nb = 3;

  testApplyColumnPermutations<Device::CPU, TypeParam>(n, nb);
}

TYPED_TEST(PermutationsTestCPU, ApplyRowPermutations) {
  SizeType n = 10;
  SizeType nb = 3;

  testApplyRowPermutations<Device::CPU, TypeParam>(n, nb);
}

TYPED_TEST(PermutationsTestCPU, Columns) {
  SizeType n = 10;
  SizeType nb = 3;
  testPermutations<Backend::MC, Device::CPU, TypeParam, Coord::Col>(n, nb);
}

TYPED_TEST(PermutationsTestCPU, Rows) {
  SizeType n = 10;
  SizeType nb = 3;
  testPermutations<Backend::MC, Device::CPU, TypeParam, Coord::Row>(n, nb);
}

#ifdef DLAF_WITH_CUDA
TYPED_TEST(PermutationsTestGPU, Columns) {
  SizeType n = 10;
  SizeType nb = 3;
  testPermutations<Backend::GPU, Device::GPU, TypeParam, Coord::Col>(n, nb);
}

TYPED_TEST(PermutationsTestGPU, Rows) {
  SizeType n = 10;
  SizeType nb = 3;
  testPermutations<Backend::GPU, Device::GPU, TypeParam, Coord::Row>(n, nb);
}
#endif
