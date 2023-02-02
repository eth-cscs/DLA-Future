//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include <gtest/gtest.h>
#include <pika/runtime.hpp>

#include "dlaf/communication/kernels.h"
#include "dlaf/eigensolver/tridiag_solver.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/multiplication/general.h"

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
using pika::this_thread::experimental::sync_wait;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <class T>
class TridiagSolverDistTestMC : public TestWithCommGrids {};
TYPED_TEST_SUITE(TridiagSolverDistTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <typename Type>
class TridiagSolverDistTestGPU : public TestWithCommGrids {};
TYPED_TEST_SUITE(TridiagSolverDistTestGPU, MatrixElementTypes);
#endif

// clang-format off
const std::vector<std::tuple<SizeType, SizeType>> tested_problems = {
    // n, nb
    {0, 8},
    {16, 16},
    {16, 8},
    {16, 4},
    {21, 4},
    {93, 7},
};
// clang-format on

// To reproduce in python:
//
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
  namespace ex = pika::execution::experimental;

  // Clone communicator to make sure non-blocking broadcasts used below don't interleave with collective
  // communication inside the tridiagonal solver.
  common::Pipeline<comm::Communicator> col_task_chain(grid.colCommunicator());
  Matrix<SizeType, Device::CPU> sign_mat(dist_evals);
  comm::Index2D this_rank = dist_evecs.rankIndex();

  // Iterate over the first tiles row
  for (SizeType j_tile = 0; j_tile < dist_evecs.nrTiles().cols(); ++j_tile) {
    GlobalTileIndex idx_evecs_0row_tile(0, j_tile);
    GlobalTileIndex idx_sign_tile(j_tile, 0);
    comm::Index2D rank_evecs_0row = dist_evecs.rankGlobalTile(idx_evecs_0row_tile);
    if (rank_evecs_0row == this_rank) {
      // If the tile is in this rank, check if signs are matching and broadcast them along the column
      SizeType j_gl_el = dist_evecs.globalElementFromGlobalTileAndTileElement<Coord::Col>(j_tile, 0);
      auto evecs_tile = sync_wait(evecs.readwrite_sender_tile(idx_evecs_0row_tile));
      auto sign_tile = sync_wait(sign_mat.readwrite_sender_tile(idx_sign_tile));

      // Iterate over the first column of the tile
      for (SizeType j_el = 0; j_el < evecs_tile.size().cols(); ++j_el) {
        GlobalElementIndex idx_gl_el(0, j_gl_el + j_el);
        TileElementIndex idx_el(0, j_el);
        auto act_val = std::real(evecs_tile(idx_el));
        auto exp_val = std::real(expected_evecs_fn(idx_gl_el));
        // If the signs of expected and actual don't match, mark the corresponding column for negation
        sign_tile(transposed(idx_el)) = (dlaf::util::sameSign(act_val, exp_val)) ? 1 : -1;
      }

      ex::start_detached(
          comm::scheduleSendBcast(ex::make_unique_any_sender(col_task_chain()),
                                  ex::make_unique_any_sender(sign_mat.read_sender2(idx_sign_tile))));
    }
    else if (rank_evecs_0row.col() == this_rank.col()) {
      // Receive signs from top column rank
      ex::start_detached(comm::scheduleRecvBcast(ex::make_unique_any_sender(col_task_chain()),
                                                 rank_evecs_0row.row(),
                                                 ex::make_unique_any_sender(
                                                     sign_mat.readwrite_sender_tile(idx_sign_tile))));
    }
  }

  for (auto idx_tile_evecs : common::iterate_range2d(dist_evecs.localNrTiles())) {
    GlobalTileIndex idx_tile_sign(dist_evecs.globalTileFromLocalTile<Coord::Col>(idx_tile_evecs.col()),
                                  0);
    auto evecs_tile = sync_wait(evecs.readwrite_sender_tile(idx_tile_evecs));
    auto sign_tile = sync_wait(sign_mat.readwrite_sender_tile(idx_tile_sign));

    for (SizeType i_el = 0; i_el < evecs_tile.size().cols(); ++i_el) {
      TileElementIndex idx_el_sign(i_el, 0);
      if (sign_tile(idx_el_sign) != -1)
        continue;
      tile::internal::scaleCol(T(-1), i_el, evecs_tile);
    }
  }

  CHECK_MATRIX_NEAR(expected_evecs_fn, evecs, complex_error * n, complex_error * n);
}

template <Backend B, Device D, class T>
void solveRandomTridiagMatrix(comm::CommunicatorGrid grid, SizeType n, SizeType nb) {
  using RealParam = BaseType<T>;

  Index2D src_rank_index(std::max(0, grid.size().rows() - 1), std::min(1, grid.size().cols() - 1));

  Distribution dist_trd(LocalElementSize(n, 2), TileElementSize(nb, 2));
  Distribution dist_evals(LocalElementSize(n, 1), TileElementSize(nb, 1));
  Distribution dist_evecs(GlobalElementSize(n, n), TileElementSize(nb, nb), grid.size(), grid.rank(),
                          src_rank_index);

  // Allocate the tridiagonal, eigenvalues and eigenvectors matrices
  Matrix<RealParam, Device::CPU> tridiag(dist_trd);
  Matrix<RealParam, Device::CPU> evals(dist_evals);
  Matrix<T, Device::CPU> evecs(dist_evecs);

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
    matrix::MatrixMirror<RealParam, D, Device::CPU> tridiag_mirror(tridiag);
    matrix::MatrixMirror<RealParam, D, Device::CPU> evals_mirror(evals);
    matrix::MatrixMirror<T, D, Device::CPU> evecs_mirror(evecs);

    // Find eigenvalues and eigenvectors of the tridiagonal matrix.
    //
    // Note: this modifies `tridiag`
    eigensolver::tridiagSolver<B>(grid, tridiag_mirror.get(), evals_mirror.get(), evecs_mirror.get());
  }

  if (n == 0)
    return;

  // Check correctness with the following equation:
  //
  // A * E = E * D, where
  //
  // A - the tridiagonal matrix
  // E - the eigenvector matrix
  // D - the diagonal matrix of eigenvalues

  // Make a copy of the tridiagonal matrix but with explicit zeroes.
  matrix::Matrix<T, Device::CPU> tridiag_full(dist_evecs);
  dlaf::matrix::util::set(tridiag_full, [&diag_arr, &offdiag_arr](GlobalElementIndex i) {
    if (i.row() == i.col()) {
      return T(diag_arr[to_sizet(i.row())]);
    }
    else if (i.row() == i.col() - 1) {
      return T(offdiag_arr[to_sizet(i.row())]);
    }
    else if (i.row() == i.col() + 1) {
      return T(offdiag_arr[to_sizet(i.col())]);
    }
    else {
      return T(0);
    }
  });
  tridiag_full.waitLocalTiles();  // makes sure that diag_arr and offdiag_arr don't go out of scope

  // To prevent creating new pipeline objects on existing communicators inside `generalSubMatrix()` which
  // my hang the unit test, clone the communicators first and then create new pipeline objects
  common::Pipeline<comm::Communicator> row_task_chain(grid.rowCommunicator().clone());
  common::Pipeline<comm::Communicator> col_task_chain(grid.colCommunicator().clone());

  // Compute A * E
  matrix::Matrix<T, Device::CPU> AE_gemm(dist_evecs);
  dlaf::multiplication::generalSubMatrix<Backend::MC, Device::CPU, T>(grid, row_task_chain,
                                                                      col_task_chain, 0,
                                                                      dist_evecs.nrTiles().rows() - 1,
                                                                      T(1), tridiag_full, evecs, T(0),
                                                                      AE_gemm);

  // Scale the columns of E by the corresponding eigenvalue of D to get E * D
  for (auto idx_loc_tile : common::iterate_range2d(dist_evecs.localNrTiles())) {
    auto scale_f = [](const matrix::Tile<const RealParam, Device::CPU>& evals_tile,
                      const matrix::Tile<T, Device::CPU>& evecs_tile) {
      for (auto el_idx_l : common::iterate_range2d(evecs_tile.size())) {
        evecs_tile(el_idx_l) *= evals_tile(TileElementIndex(el_idx_l.col(), 0));
      }
    };

    GlobalTileIndex idx_gl_evals(dist_evecs.globalTileFromLocalTile<Coord::Col>(idx_loc_tile.col()), 0);
    dlaf::internal::whenAllLift(evals.read_sender2(idx_gl_evals),
                                evecs.readwrite_sender_tile(idx_loc_tile)) |
        dlaf::internal::transformDetach(dlaf::internal::Policy<Backend::MC>(), std::move(scale_f));
  }

  // Check that A * E is equal to E * D
  constexpr RealParam error = TypeUtilities<T>::error;
  for (auto idx_loc_tile : common::iterate_range2d(dist_evecs.localNrTiles())) {
    CHECK_TILE_NEAR(sync_wait(AE_gemm.read_sender2(idx_loc_tile)).get(),
                    sync_wait(evecs.read_sender2(idx_loc_tile)).get(), error * n, error * n);
  }
}

TYPED_TEST(TridiagSolverDistTestMC, Laplace1D) {
  for (const auto& comm_grid : this->commGrids()) {
    for (auto [n, nb] : tested_problems) {
      solveDistributedLaplace1D<Backend::MC, Device::CPU, TypeParam>(comm_grid, n, nb);
      pika::threads::get_thread_manager().wait();
    }
  }
}

TYPED_TEST(TridiagSolverDistTestMC, Random) {
  for (const auto& comm_grid : this->commGrids()) {
    for (auto [n, nb] : tested_problems) {
      solveRandomTridiagMatrix<Backend::MC, Device::CPU, TypeParam>(comm_grid, n, nb);
      pika::threads::get_thread_manager().wait();
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(TridiagSolverDistTestGPU, Laplace1D) {
  for (const auto& comm_grid : this->commGrids()) {
    for (auto [n, nb] : tested_problems) {
      solveDistributedLaplace1D<Backend::GPU, Device::GPU, TypeParam>(comm_grid, n, nb);
      pika::threads::get_thread_manager().wait();
    }
  }
}

TYPED_TEST(TridiagSolverDistTestGPU, Random) {
  for (const auto& comm_grid : this->commGrids()) {
    for (auto [n, nb] : tested_problems) {
      solveRandomTridiagMatrix<Backend::GPU, Device::GPU, TypeParam>(comm_grid, n, nb);
      pika::threads::get_thread_manager().wait();
    }
  }
}
#endif
