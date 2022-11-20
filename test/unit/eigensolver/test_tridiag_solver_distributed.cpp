//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include <gtest/gtest.h>
#include <pika/runtime.hpp>

#include "dlaf/communication/sync/broadcast.h"
#include "dlaf/eigensolver/tridiag_solver.h"
#include "dlaf/matrix/matrix_mirror.h"

#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_generic_lapack.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <class T>
struct TridiagSolverDistTestMC : public TestWithCommGrids {};

TYPED_TEST_SUITE(TridiagSolverDistTestMC, MatrixElementTypes);

// clang-format off
const std::vector<std::tuple<SizeType, SizeType>> tested_problems = {
    // n, nb
//    {0, 8},
//    {16, 16},
    {16, 8},
//    {16, 4},
 //   {16, 5},
 //   {100, 10},
 //   {93, 7},
};
// clang-format on

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
void solveDistributedLaplace1D(comm::CommunicatorGrid grid, SizeType n, SizeType nb) {
  // namespace ex = pika::execution::experimental;
  // namespace tt = pika::this_thread::experimental;

  using RealParam = BaseType<T>;
  constexpr RealParam complex_error = TypeUtilities<T>::error;
  constexpr RealParam real_error = TypeUtilities<RealParam>::error;

  Index2D src_rank_index(std::max(0, grid.size().rows() - 1), std::min(1, grid.size().cols() - 1));

  Distribution dist_trd(LocalElementSize(n, 2), TileElementSize(nb, 2));
  Distribution dist_evals(LocalElementSize(n, 1), TileElementSize(nb, 1));
  Distribution dist_evecs(GlobalElementSize(n, n), TileElementSize(nb, nb), grid.size(), grid.rank(),
                          src_rank_index);

  Matrix<RealParam, Device::CPU> tridiag(dist_trd);
  Matrix<RealParam, Device::CPU> evals(dist_evals);
  Matrix<T, Device::CPU> evecs(dist_evecs);

  // Tridiagonal matrix : 1D Laplacian
  auto mat_trd_fn = [](GlobalElementIndex el) {
    if (el.col() == 0)
      // diagonal
      return RealParam(2);
    else
      // off-diagonal
      return RealParam(-1);
  };
  matrix::util::set(tridiag, std::move(mat_trd_fn));

  {
    matrix::MatrixMirror<RealParam, D, Device::CPU> tridiag_mirror(tridiag);
    matrix::MatrixMirror<RealParam, D, Device::CPU> evals_mirror(evals);
    matrix::MatrixMirror<T, D, Device::CPU> evecs_mirror(evecs);

    eigensolver::tridiagSolver<B>(grid, tridiag_mirror.get(), evals_mirror.get(), evecs_mirror.get());
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

  // Eigenvectors are unique up to a sign, match signs of expected and actual eigenvectors
  auto col_comm = grid.colCommunicator();
  SizeType gl_row = dist_evecs.globalElementFromLocalTileAndTileElement<Coord::Row>(0, 0);
  // local tile elements that are to be negated
  Distribution dist_sign(LocalElementSize(dist_evecs.localSize().cols(), 1), TileElementSize(nb, 1));
  Matrix<SizeType, Device::CPU> sign_mat(dist_sign);

  // if global row is zero
  if (gl_row == 0) {
    for (SizeType i_tile = 0; i_tile < dist_evecs.localNrTiles().cols(); ++i_tile) {
      SizeType i_gl_el = dist_evecs.globalElementFromLocalTileAndTileElement<Coord::Col>(i_tile, 0);
      LocalTileIndex idx_tile(0, i_tile);
      auto evecs_tile = evecs(idx_tile).get();
      auto sign_tile = sign_mat(transposed(idx_tile)).get();
      // Iterate over the first column of the tile
      for (SizeType i_el = 0; i_el < evecs_tile.size().cols(); ++i_el) {
        GlobalElementIndex idx_gl_el(0, i_gl_el + i_el);
        TileElementIndex idx_el(0, i_el);
        auto act_val = std::real(evecs_tile(idx_el));
        auto exp_val = std::real(expected_evecs_fn(idx_gl_el));
        // If the signs of expected and actual don't match, mark the corresponding column for negation
        sign_tile(transposed(idx_el)) = (dlaf::util::sameSign(act_val, exp_val)) ? 1 : -1;
      }
      sync::broadcast::send(col_comm, sign_tile);
    }
  }
  else {
    for (auto idx_tile : common::iterate_range2d(dist_sign.localNrTiles())) {
      sync::broadcast::receive_from(0, col_comm, sign_mat(idx_tile).get());
    }
  }

  for (auto idx_tile_evecs : common::iterate_range2d(dist_evecs.localNrTiles())) {
    LocalTileIndex idx_tile_sign(idx_tile_evecs.col(), 0);
    auto evecs_tile = evecs(idx_tile_evecs).get();
    auto sign_tile = sign_mat(idx_tile_sign).get();

    for (SizeType i_el = 0; i_el < evecs_tile.size().cols(); ++i_el) {
      TileElementIndex idx_el_sign(i_el, 0);
      if (sign_tile(idx_el_sign) != -1)
        continue;
      tile::internal::scaleCol(T(-1), i_el, evecs_tile);
    }
  }

  CHECK_MATRIX_NEAR(expected_evecs_fn, evecs, complex_error * n, complex_error * n);
}

TEST(TridiagSolverDistTestMC, Laplace1D) {
  using TypeParam = float;
  //CommunicatorGrid comm_grid(MPI_COMM_WORLD, 2, 3, common::Ordering::ColumnMajor);
  CommunicatorGrid comm_grid(MPI_COMM_WORLD, 2, 3, common::Ordering::ColumnMajor);
  // for (const auto& comm_grid : this->commGrids()) {
  for (auto [n, nb] : tested_problems) {
    solveDistributedLaplace1D<Backend::MC, Device::CPU, TypeParam>(comm_grid, n, nb);
    pika::threads::get_thread_manager().wait();
  }
  //}
}
