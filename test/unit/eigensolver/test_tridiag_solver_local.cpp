//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <utility>
#include <vector>

#include <dlaf/eigensolver/tridiag_solver.h>
#include <dlaf/eigensolver/tridiag_solver/impl.h>
#include <dlaf/matrix/matrix_mirror.h>

#include <gtest/gtest.h>

#include <dlaf_test/eigensolver/test_eigensolver_correctness.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/matrix/util_tile.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::test;
using pika::this_thread::experimental::sync_wait;

template <typename Type>
class TridiagEigensolverTestCPU : public ::testing::Test {};
TYPED_TEST_SUITE(TridiagEigensolverTestCPU, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <typename Type>
class TridiagEigensolverTestGPU : public ::testing::Test {};
TYPED_TEST_SUITE(TridiagEigensolverTestGPU, MatrixElementTypes);
#endif

TEST(MatrixIndexPairsGeneration, IndexPairsGeneration) {
  SizeType n = 10;
  auto actual_indices = dlaf::eigensolver::internal::generateSubproblemIndices(n);
  // i_begin, i_split, i_end
  std::vector<std::tuple<SizeType, SizeType, SizeType>> expected_indices{{0, 1, 2},  {0, 2, 3},
                                                                         {3, 4, 5},  {0, 3, 5},
                                                                         {5, 6, 7},  {5, 7, 8},
                                                                         {8, 9, 10}, {5, 8, 10},
                                                                         {0, 5, 10}};
  ASSERT_TRUE(actual_indices == expected_indices);
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
template <Backend B, Device D, class T>
void solveLaplace1D(SizeType n, SizeType nb) {
  using RealParam = BaseType<T>;
  constexpr RealParam complex_error = TypeUtilities<T>::error;
  constexpr RealParam real_error = TypeUtilities<RealParam>::error;

  matrix::Matrix<RealParam, Device::CPU> tridiag(LocalElementSize(n, 2), TileElementSize(nb, 2));
  matrix::Matrix<RealParam, Device::CPU> evals(LocalElementSize(n, 1), TileElementSize(nb, 1));
  matrix::Matrix<T, Device::CPU> evecs(LocalElementSize(n, n), TileElementSize(nb, nb));

  // Tridiagonal matrix : 1D Laplacian
  auto tridiag_fn = [](GlobalElementIndex el) {
    if (el.col() == 0)
      // diagonal
      return RealParam(2);
    else
      // off-diagonal
      return RealParam(-1);
  };
  matrix::util::set(tridiag, std::move(tridiag_fn));

  {
    matrix::MatrixMirror<RealParam, D, Device::CPU> evals_mirror(evals);
    matrix::MatrixMirror<T, D, Device::CPU> evecs_mirror(evecs);

    eigensolver::internal::tridiagonal_eigensolver<B>(tridiag, evals_mirror.get(), evecs_mirror.get());
  }
  if (n == 0)
    return;

  // Eigenvalues
  auto expected_evals_fn = [n](GlobalElementIndex i) {
    return RealParam(2 * (1 - std::cos(M_PI * (i.row() + 1) / (n + 1))));
  };
  CHECK_MATRIX_NEAR(expected_evals_fn, evals, n * real_error, n * real_error);

  // Eigenvectors
  auto expected_evecs_fn = [n](GlobalElementIndex i) {
    SizeType j = i.col() + 1;
    SizeType k = i.row() + 1;
    return TypeUtilities<T>::element(std::sqrt(2.0 / (n + 1)) * std::sin(j * k * M_PI / (n + 1)), 0);
  };

  // Eigenvectors are unique up to a sign
  std::vector<SizeType> neg_cols;  // columns to negate
  neg_cols.reserve(to_sizet(n));
  const auto& dist = evecs.distribution();
  for (SizeType i_tile = 0; i_tile < dist.nrTiles().cols(); ++i_tile) {
    SizeType i_gl_el = dist.template globalElementFromGlobalTileAndTileElement<Coord::Col>(i_tile, 0);
    auto tile = sync_wait(evecs.readwrite(GlobalTileIndex(0, i_tile)));
    for (SizeType i_tile_el = 0; i_tile_el < tile.size().cols(); ++i_tile_el) {
      if (dlaf::util::sameSign(std::real(expected_evecs_fn(GlobalElementIndex(0, i_gl_el + i_tile_el))),
                               std::real(tile(TileElementIndex(0, i_tile_el)))))
        continue;
      neg_cols.push_back(i_gl_el + i_tile_el);
    }
  }

  for (SizeType i_gl_el : neg_cols) {
    SizeType j_tile = dist.template globalTileFromGlobalElement<Coord::Col>(i_gl_el);
    SizeType j_tile_el = dist.template tileElementFromGlobalElement<Coord::Col>(i_gl_el);

    // Iterate over all tiles on the `j_tile` tile column
    for (SizeType i_tile = 0; i_tile < dist.nrTiles().rows(); ++i_tile) {
      auto tile = sync_wait(evecs.readwrite(GlobalTileIndex(i_tile, j_tile)));
      tile::internal::scaleCol(T(-1), j_tile_el, tile);
    }
  }

  CHECK_MATRIX_NEAR(expected_evecs_fn, evecs, complex_error * n, complex_error * n);
}

template <Backend B, Device D, class T>
void solveRandomTridiagMatrix(SizeType n, SizeType nb) {
  using RealParam = BaseType<T>;

  // Allocate the tridiagonal, eigenvalues and eigenvectors matrices
  matrix::Matrix<RealParam, Device::CPU> tridiag(LocalElementSize(n, 2), TileElementSize(nb, 2));
  matrix::Matrix<RealParam, Device::CPU> evals(LocalElementSize(n, 1), TileElementSize(nb, 1));
  matrix::Matrix<T, Device::CPU> evecs(LocalElementSize(n, n), TileElementSize(nb, nb));

  // Initialize a random symmetric tridiagonal matrix using two arrays for the diagonal and the
  // off-diagonal. The random numbers are in the range [-1, 1].
  //
  // Note: set_random() is not used here because the two arrays can be more easily reused to initialize
  //       the same tridiagonal matrix but with explicit zeros for correctness checking further down.
  std::vector<RealParam> diag_arr(to_sizet(n));
  std::vector<RealParam> offdiag_arr(to_sizet(n));
  SizeType diag_seed = n;
  SizeType offdiag_seed = n + 1;
  dlaf::matrix::util::internal::getter_random<RealParam> diag_rand_gen(diag_seed);
  dlaf::matrix::util::internal::getter_random<RealParam> offdiag_rand_gen(offdiag_seed);
  std::generate(std::begin(diag_arr), std::end(diag_arr), diag_rand_gen);
  std::generate(std::begin(offdiag_arr), std::end(offdiag_arr), offdiag_rand_gen);

  dlaf::matrix::util::set(tridiag, [&diag_arr, &offdiag_arr](GlobalElementIndex i) {
    if (i.col() == 0) {
      return diag_arr[to_sizet(i.row())];
    }
    else {
      return offdiag_arr[to_sizet(i.row())];
    }
  });
  tridiag.waitLocalTiles();  // makes sure that diag_arr and offdiag_arr don't go out of scope

  {
    matrix::MatrixMirror<RealParam, D, Device::CPU> evals_mirror(evals);
    matrix::MatrixMirror<T, D, Device::CPU> evecs_mirror(evecs);

    // Find eigenvalues and eigenvectors of the tridiagonal matrix.
    //
    // Note: this modifies `tridiag`
    eigensolver::internal::tridiagonal_eigensolver<B>(tridiag, evals_mirror.get(), evecs_mirror.get());
  }

  if (n == 0)
    return;

  // Check correctness

  // Make a copy of the tridiagonal matrix (Lower) but with explicit zeroes.
  matrix::Matrix<T, Device::CPU> tridiag_full(LocalElementSize(n, n), TileElementSize(nb, nb));
  dlaf::matrix::util::set(tridiag_full, [&diag_arr, &offdiag_arr](GlobalElementIndex i) {
    if (i.row() == i.col()) {
      return T(diag_arr[to_sizet(i.row())]);
    }
    else if (i.row() == i.col() + 1) {
      return T(offdiag_arr[to_sizet(i.col())]);
    }
    else {
      return T(0);
    }
  });
  tridiag_full.waitLocalTiles();  // makes sure that diag_arr and offdiag_arr don't go out of scope

  testEigensolverCorrectness(blas::Uplo::Lower, tridiag_full, evals, evecs);
}

// clang-format off
const std::vector<std::tuple<SizeType, SizeType>> tested_problems = {
    // n, nb
    {0, 8},
    {4, 2},
    {16, 16},
    {16, 8},
    {16, 4},
    {16, 5},
    {100, 10},
    {93, 7},
};
// clang-format on

TYPED_TEST(TridiagEigensolverTestCPU, Laplace1D) {
  for (auto [n, nb] : tested_problems) {
    solveLaplace1D<Backend::MC, Device::CPU, TypeParam>(n, nb);
  }
}

TYPED_TEST(TridiagEigensolverTestCPU, Random) {
  for (auto [n, nb] : tested_problems) {
    solveRandomTridiagMatrix<Backend::MC, Device::CPU, TypeParam>(n, nb);
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(TridiagEigensolverTestGPU, Laplace1D) {
  for (auto [n, nb] : tested_problems) {
    solveLaplace1D<Backend::GPU, Device::GPU, TypeParam>(n, nb);
  }
}

TYPED_TEST(TridiagEigensolverTestGPU, Random) {
  for (auto [n, nb] : tested_problems) {
    solveRandomTridiagMatrix<Backend::GPU, Device::GPU, TypeParam>(n, nb);
  }
}
#endif
