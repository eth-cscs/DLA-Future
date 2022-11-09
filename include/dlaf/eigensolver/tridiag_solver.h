//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include "dlaf/common/assert.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/eigensolver/tridiag_solver/impl.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {

/// Finds the eigenvalues and eigenvectors of a symmetric tridiagonal matrix.
///
/// @param mat_trd  [in/out] (n x 2) local matrix with the diagonal and off-diagonal of the symmetric
/// tridiagonal matrix in the first column and second columns respectively. The last entry of the second
/// column is not used. On exit the eigenvalues are saved in the first column.
//
/// @param d [out] (n x 1) local matrix holding the eigenvalues of the the symmetric tridiagonal mat_trd
//
/// @param mat_ev [out]    (n x n) local matrix holding the eigenvectors of the the symmetric tridiagonal
/// matrix on exit.
///
/// @pre mat_trd and mat_ev are local matrices
/// @pre mat_trd has 2 columns
/// @pre mat_ev is a square matrix
/// @pre mat_ev has a square block size
template <Backend backend, Device device, class T>
void tridiagSolver(Matrix<BaseType<T>, device>& tridiag, Matrix<BaseType<T>, device>& evals,
                   Matrix<T, device>& evecs) {
  DLAF_ASSERT(matrix::local_matrix(tridiag), tridiag);
  DLAF_ASSERT(tridiag.distribution().size().cols() == 2, tridiag);

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

#if defined(DLAF_WITH_HIP)
  DLAF_UNIMPLEMENTED("Tridiagonal solver is not yet implemented for HIP");
#else
  internal::TridiagSolver<backend, device, BaseType<T>>::call(tridiag, evals, evecs);
#endif
}

}
}
