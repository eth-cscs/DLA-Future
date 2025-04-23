//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

/// @file

#include <dlaf/common/assert.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/eigensolver/tridiag_solver/api.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace dlaf::eigensolver::internal {

/// Finds the eigenvalues and eigenvectors of the local symmetric tridiagonal matrix @p tridiag.
///
/// @param tridiag local matrix with the diagonal and off-diagonal of the symmetric tridiagonal
///                matrix in the first column and second columns respectively. The last entry of the
///                second column is not used.
/// @pre @p tridiag is not distributed
/// @pre @p tridiag has size (N x 2)
/// @pre @p tridiag has block size (NB x 2)
/// @pre @p tridiag has tile size (NB x 2)
///
/// @param[out] evals contains the eigenvalues of the symmetric tridiagonal matrix
/// @pre @p evals is not distributed
/// @pre @p evals has size (N x 1)
/// @pre @p evals has block size (NB x 1)
/// @pre @p evals has tile size (NB x 1)
///
/// @param[out] evecs contains the eigenvectors of the symmetric tridiagonal matrix
/// @pre @p evecs is not distributed
/// @pre @p evecs has size (N x N)
/// @pre @p evecs has block size (NB x NB)
/// @pre @p evecs has tile size (NB x NB)
template <Backend backend, Device device, class T>
void tridiagonal_eigensolver(Matrix<BaseType<T>, Device::CPU>& tridiag,
                             Matrix<BaseType<T>, device>& evals, Matrix<T, device>& evecs) {
  DLAF_ASSERT(matrix::local_matrix(tridiag), tridiag);
  DLAF_ASSERT(tridiag.size().cols() == 2, tridiag);
  DLAF_ASSERT(tridiag.block_size().cols() == 2, tridiag);
  DLAF_ASSERT(matrix::single_tile_per_block(tridiag), tridiag);

  DLAF_ASSERT(matrix::local_matrix(evals), evals);
  DLAF_ASSERT(evals.size().cols() == 1, evals);

  DLAF_ASSERT(matrix::local_matrix(evecs), evecs);
  DLAF_ASSERT(matrix::square_size(evecs), evecs);
  DLAF_ASSERT(matrix::square_block_size(evecs), evecs);

  DLAF_ASSERT(matrix::single_tile_per_block(evecs), evecs);
  DLAF_ASSERT(matrix::single_tile_per_block(evals), evals);

  DLAF_ASSERT(tridiag.block_size().rows() == evecs.block_size().rows(), evecs.block_size().rows(),
              tridiag.block_size().rows());
  DLAF_ASSERT(tridiag.block_size().rows() == evals.block_size().rows(), tridiag.block_size().rows(),
              evals.block_size().rows());
  DLAF_ASSERT(tridiag.size().rows() == evecs.size().rows(), evecs.size().rows(), tridiag.size().rows());
  DLAF_ASSERT(tridiag.size().rows() == evals.size().rows(), tridiag.size().rows(), evals.size().rows());

  TridiagSolver<backend, device, BaseType<T>>::call(tridiag, evals, evecs);
}

/// Finds the eigenvalues and eigenvectors of the symmetric tridiagonal matrix @p tridiag stored locally
/// on each rank. The resulting eigenvalues @p evals are stored locally on each rank while the resulting
/// eigenvectors @p evecs are distributed across ranks in 2D block-cyclic manner.
///
/// @param tridiag matrix with the diagonal and off-diagonal of the symmetric tridiagonal matrix in the
///                first column and second columns respectively. The last entry of the second column is
///                not used.
/// @pre @p tridiag is not distributed
/// @pre @p tridiag has size (N x 2)
/// @pre @p tridiag has block size (NB x 2)
/// @pre @p tridiag has tile size (NB x 2)
///
/// @param[out] evals holds the eigenvalues of the symmetric tridiagonal matrix
/// @pre @p evals is not distributed
/// @pre @p evals has size (N x 1)
/// @pre @p evals has block size (NB x 1)
/// @pre @p evals has tile size (NB x 1)
///
/// @param[out] evecs holds the eigenvectors of the symmetric tridiagonal matrix
/// @pre @p evecs is distributed according to @p grid
/// @pre @p evecs has size (N x N)
/// @pre @p evecs has block size (NB x NB)
/// @pre @p evecs has tile size (NB x NB)
template <Backend B, Device D, class T>
void tridiagonal_eigensolver(comm::CommunicatorGrid& grid, Matrix<BaseType<T>, Device::CPU>& tridiag,
                             Matrix<BaseType<T>, D>& evals, Matrix<T, D>& evecs) {
  DLAF_ASSERT(matrix::local_matrix(tridiag), tridiag);
  DLAF_ASSERT(tridiag.size().cols() == 2, tridiag);
  DLAF_ASSERT(tridiag.block_size().cols() == 2, tridiag);
  DLAF_ASSERT(matrix::single_tile_per_block(tridiag), tridiag);

  DLAF_ASSERT(matrix::local_matrix(evals), evals);
  DLAF_ASSERT(evals.size().cols() == 1, evals);

  DLAF_ASSERT(matrix::square_size(evecs), evecs);
  DLAF_ASSERT(matrix::square_block_size(evecs), evecs);
  DLAF_ASSERT(matrix::equal_process_grid(evecs, grid), evecs, grid);

  DLAF_ASSERT(matrix::single_tile_per_block(evecs), evecs);
  DLAF_ASSERT(matrix::single_tile_per_block(evals), evals);

  DLAF_ASSERT(tridiag.block_size().rows() == evecs.block_size().rows(), evecs, tridiag);
  DLAF_ASSERT(tridiag.block_size().rows() == evals.block_size().rows(), tridiag, evals);
  DLAF_ASSERT(tridiag.size().rows() == evecs.size().rows(), evecs, tridiag);
  DLAF_ASSERT(tridiag.size().rows() == evals.size().rows(), tridiag, evals);

  TridiagSolver<B, D, BaseType<T>>::call(grid, tridiag, evals, evecs);
}

}
