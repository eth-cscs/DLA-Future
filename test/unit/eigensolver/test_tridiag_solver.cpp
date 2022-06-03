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
#include "dlaf_test/matrix/util_matrix.h"
//#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

#include "dlaf/matrix/print_csv.h"

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
  expected_top(TileElementIndex(sz - 1, 0)) = TypeParam(1);
  expected_bottom(TileElementIndex(0, 0)) = TypeParam(1);

  CHECK_TILE_NEAR(expected_top, top, TypeUtilities<TypeParam>::error, TypeUtilities<TypeParam>::error);
  CHECK_TILE_NEAR(expected_bottom, bottom, TypeUtilities<TypeParam>::error,
                  TypeUtilities<TypeParam>::error);
}

// import numpy as np
// from scipy.sparse import diags
// from scipy.linalg import eigh
//
// n = 10
// d = np.full(n, 2)
// e = np.full(n - 1, -1)
// trd = diags([e,d,e], [-1, 0, 1]).toarray()
// evals, evecs = eigh(trd)
//

template <class T>
void solveLaplace1D(SizeType n, SizeType nb) {
  constexpr double pi = 3.14159265358979323846;

  matrix::Matrix<T, Device::CPU> tridiag(LocalElementSize(n, 2), TileElementSize(nb, 2));
  matrix::Matrix<T, Device::CPU> evals(LocalElementSize(n, 1), TileElementSize(nb, 1));
  matrix::Matrix<T, Device::CPU> evecs(LocalElementSize(n, n), TileElementSize(nb, nb));

  // Tridiagonal matrix : 1D Laplacian
  auto mat_trd_fn = [](GlobalElementIndex el) {
    if (el.col() == 0)
      // diagonal
      return T(2);
    else
      // off-diagoanl
      return T(-1);
  };
  matrix::util::set(tridiag, std::move(mat_trd_fn));

  eigensolver::tridiagSolver<Backend::MC>(tridiag, evals, evecs);

  // Eigenvalues
  auto expected_evals_fn = [n](GlobalElementIndex i) {
    return T(2 * (1 - std::cos(pi * (i.row() + 1) / (n + 1))));
  };
  CHECK_MATRIX_NEAR(expected_evals_fn, evals, 1e-6, 1e-6);

  // Eigenvectors
  auto expected_evecs_fn = [n](GlobalElementIndex i) {
    SizeType j = i.col() + 1;
    SizeType k = i.row() + 1;
    return T(std::sqrt(2.0 / (n + 1)) * std::sin(j * k * pi / (n + 1)));
  };

  // Eigenvectors are unique up to a sign
  std::vector<SizeType> neg_cols;  // columns to negate
  neg_cols.reserve(to_sizet(n));
  const auto& dist = evecs.distribution();
  for (SizeType i_tile = 0; i_tile < dist.nrTiles().cols(); ++i_tile) {
    SizeType i_gl_el = dist.template globalElementFromGlobalTileAndTileElement<Coord::Col>(i_tile, 0);
    auto tile = evecs(GlobalTileIndex(0, i_tile)).get();
    for (SizeType i_tile_el = 0; i_tile_el < tile.size().cols(); ++i_tile_el) {
      if (dlaf::util::sameSign(expected_evecs_fn(GlobalElementIndex(0, i_gl_el + i_tile_el)),
                               tile(TileElementIndex(0, i_tile_el))))
        continue;
      neg_cols.push_back(i_gl_el + i_tile_el);
    }
  }

  for (SizeType i_gl_el : neg_cols) {
    SizeType j_tile = dist.template globalTileFromGlobalElement<Coord::Col>(i_gl_el);
    SizeType j_tile_el = dist.template tileElementFromGlobalElement<Coord::Col>(i_gl_el);

    // Iterate over all tiles on the `j_tile` tile column
    for (SizeType i_tile = 0; i_tile < dist.nrTiles().rows(); ++i_tile) {
      auto tile = evecs(GlobalTileIndex(i_tile, j_tile)).get();
      tile::internal::scaleCol(T(-1), j_tile_el, tile);
    }
  }
  CHECK_MATRIX_NEAR(expected_evecs_fn, evecs, 1e-6, 1e-6);
}

TYPED_TEST(TridiagEigensolverTest, Laplace1D_n16_nb8) {
  solveLaplace1D<TypeParam>(16, 8);
}

TYPED_TEST(TridiagEigensolverTest, Laplace1D_n16_nb4) {
  solveLaplace1D<TypeParam>(16, 4);
}

// This occasionally segfaults. It may also deadlock?
// TYPED_TEST(TridiagEigensolverTest, Laplace1D_n16_nb5) {
//  solveLaplace1D<TypeParam>(16, 5);
//}
