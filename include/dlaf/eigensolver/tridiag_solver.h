//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

/// @file

#include "dlaf/common/assert.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/eigensolver/tridiag_solver/api.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {

/// Finds the eigenvalues and eigenvectors of the local symmetric tridiagonal matrix @p tridiag.
///
/// @param tridiag [in/out] (n x 2) local matrix with the diagonal and off-diagonal of the symmetric
///                tridiagonal matrix in the first column and second columns respectively. The last entry
///                of the second column is not used.
/// @param evals [out] (n x 1) local matrix holding the eigenvalues of the the symmetric tridiagonal
///              matrix
/// @param evecs [out] (n x n) local matrix holding the eigenvectors of the the symmetric tridiagonal
///              matrix on exit.
///
/// @pre tridiag and @p evals and @p evecs are local matrices
/// @pre tridiag has 2 columns and column block size of 2
/// @pre evecs is a square matrix with number of rows equal to the number of rows of @p tridiag and @p evals
/// @pre evecs has a square block size with number of block rows eqaul to the block rows of @p tridiag and @p evals
template <Backend backend, Device device, class T>
void tridiagSolver(Matrix<BaseType<T>, Device::CPU>& tridiag, Matrix<BaseType<T>, device>& evals,
                   Matrix<T, device>& evecs) {
  DLAF_ASSERT(matrix::local_matrix(tridiag), tridiag);
  DLAF_ASSERT(tridiag.distribution().size().cols() == 2, tridiag);
  DLAF_ASSERT(tridiag.distribution().blockSize().cols() == 2, tridiag);

  DLAF_ASSERT(matrix::local_matrix(evals), evals);
  DLAF_ASSERT(evals.distribution().size().cols() == 1, evals);

  DLAF_ASSERT(matrix::local_matrix(evecs), evecs);
  DLAF_ASSERT(matrix::square_size(evecs), evecs);
  DLAF_ASSERT(matrix::square_blocksize(evecs), evecs);

  DLAF_ASSERT(tridiag.distribution().blockSize().rows() == evecs.distribution().blockSize().rows(),
              evecs.distribution().blockSize().rows(), tridiag.distribution().blockSize().rows());
  DLAF_ASSERT(tridiag.distribution().blockSize().rows() == evals.distribution().blockSize().rows(),
              tridiag.distribution().blockSize().rows(), evals.distribution().blockSize().rows());
  DLAF_ASSERT(tridiag.distribution().size().rows() == evecs.distribution().size().rows(),
              evecs.distribution().size().rows(), tridiag.distribution().size().rows());
  DLAF_ASSERT(tridiag.distribution().size().rows() == evals.distribution().size().rows(),
              tridiag.distribution().size().rows(), evals.distribution().size().rows());

  internal::TridiagSolver<backend, device, BaseType<T>>::call(tridiag, evals, evecs);
}

/// Finds the eigenvalues and eigenvectors of the symmetric tridiagonal matrix @p tridiag stored locally
/// on each rank. The resulting eigenvalues @p evals are stored locally on each rank while the resulting
/// eigenvectors @p evecs are distributed across ranks in 2D block-cyclic manner.
///
/// @param tridiag [in/out] (n x 2) local matrix with the diagonal and off-diagonal of the symmetric
///                tridiagonal matrix in the first column and second columns respectively. The last entry
///                of the second column is not used.
/// @param evals [out] (n x 1) local matrix holding the eigenvalues of the the symmetric tridiagonal
///              matrix
/// @param evecs [out] (n x n) distributed matrix holding the eigenvectors of the the symmetric tridiagonal
///              matrix on exit.
///
/// @pre tridiag and @p evals are local matrices and are the same on all ranks
/// @pre tridiag has 2 columns and column block size of 2
/// @pre evecs is a square matrix with global number of rows equal to the number of rows of @p tridiag and @p evals
/// @pre evecs has a square block size with number of block rows eqaul to the block rows of @p tridiag and @p evals
template <Backend B, Device D, class T>
void tridiagSolver(comm::CommunicatorGrid grid, Matrix<BaseType<T>, Device::CPU>& tridiag,
                   Matrix<BaseType<T>, D>& evals, Matrix<T, D>& evecs) {
  DLAF_ASSERT(matrix::local_matrix(tridiag), tridiag);
  DLAF_ASSERT(tridiag.distribution().size().cols() == 2, tridiag);
  DLAF_ASSERT(tridiag.distribution().blockSize().cols() == 2, tridiag);

  DLAF_ASSERT(matrix::local_matrix(evals), evals);
  DLAF_ASSERT(evals.distribution().size().cols() == 1, evals);

  DLAF_ASSERT(matrix::square_size(evecs), evecs);
  DLAF_ASSERT(matrix::square_blocksize(evecs), evecs);
  DLAF_ASSERT(matrix::equal_process_grid(evecs, grid), evecs, grid);

  DLAF_ASSERT(tridiag.distribution().blockSize().rows() == evecs.distribution().blockSize().rows(),
              evecs, tridiag);
  DLAF_ASSERT(tridiag.distribution().blockSize().rows() == evals.distribution().blockSize().rows(),
              tridiag, evals);
  DLAF_ASSERT(tridiag.distribution().size().rows() == evecs.distribution().size().rows(), evecs,
              tridiag);
  DLAF_ASSERT(tridiag.distribution().size().rows() == evals.distribution().size().rows(), tridiag,
              evals);

  internal::TridiagSolver<B, D, BaseType<T>>::call(grid, tridiag, evals, evecs);
}

}
}
