//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/eigensolver/tridiag_solver/api.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {

/// Finds the eigenvalues and eigenvectors of a symmetric tridiagonal matrix.
///
/// @param mat_a  [in/out] `n x 2` matrix with the diagonal and off-diagonal of the symmetric tridiagonal
/// matrix in the first column and second columns respectively. The last entry of the second column is
/// not used. On exit the eigenvalues are saved in the first column.
/// @param mat_ev [out]    `n x n` matrix holding the eigenvectors of the the symmetric tridiagonal
/// matrix on exit.
///
/// @pre mat_a and mat_ev are local matrices
/// @pre mat_a has 2 columns
/// @pre mat_ev is a square matrix
/// @pre mat_ev has a square block size
template <Backend backend, Device device, class T>
void tridiagSolver(Matrix<T, device>& mat_a, Matrix<T, device>& mat_ev) {
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);
  DLAF_ASSERT(mat_a.distribution().size().cols() == 2, mat_a);

  DLAF_ASSERT(matrix::local_matrix(mat_ev), mat_ev);
  DLAF_ASSERT(matrix::square_size(mat_ev), mat_ev);
  DLAF_ASSERT(matrix::square_blocksize(mat_ev), mat_ev);

  internal::TridiagSolver<backend, device, T>::call(mat_a, mat_ev);
}

/// TODO: more info on the distributed version
/// Finds the eigenvalues and eigenvectors of a symmetric tridiagonal matrix.
///
/// @param mat_a  [in/out] `n x 2` matrix with the diagonal and off-diagonal of the symmetric tridiagonal
/// matrix in the first column and second columns respectively. The last entry of the second column is
/// not used. On exit the eigenvalues are saved in the first column.
/// @param mat_ev [out]    `n x n` matrix holding the eigenvectors of the the symmetric tridiagonal
/// matrix on exit.
///
/// @pre mat_a and mat_ev are local matrices
/// @pre mat_a has 2 columns
/// @pre mat_ev is a square matrix
/// @pre mat_ev has a square block size
template <Backend backend, Device device, class T>
void tridiagSolver(comm::CommunicatorGrid grid, Matrix<T, device>& mat_a, Matrix<T, device>& mat_ev) {
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);
  DLAF_ASSERT(mat_a.distribution().size().cols() == 2, mat_a);

  DLAF_ASSERT(matrix::local_matrix(mat_ev), mat_ev);
  DLAF_ASSERT(matrix::square_size(mat_ev), mat_ev);
  DLAF_ASSERT(matrix::square_blocksize(mat_ev), mat_ev);
  DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);
  DLAF_ASSERT(matrix::equal_process_grid(mat_ev, grid), mat_ev, grid);

  internal::TridiagSolver<backend, device, T>::call(std::move(grid), mat_a, mat_ev);
}

}
}
