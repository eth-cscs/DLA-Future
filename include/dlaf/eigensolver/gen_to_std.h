//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

/// @file

#include <blas.hh>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/eigensolver/gen_to_std/api.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace dlaf::eigensolver::internal {

/// Reduce a Hermitian definite generalized eigenproblem to standard form.
///
/// It computes inv(B)*A*inv(B**H) or inv(B**H)*A*inv(B), where B have been previously factorized
/// using the Cholesky factorization.
/// Implementation on local memory.
///
/// @param uplo specifies if the elements of the Hermitian matrix A and the triangular matrix B
/// to be referenced are the elements in the lower or upper triangular part,
///
/// @param mat_a on entry it contains the Hermitian matrix A (if A is real, the matrix is symmetric),
/// on exit the matrix elements are overwritten with the elements of the matrix B.
/// Only the tiles of the matrix which contain the lower triangular or the upper triangular part are accessed.
/// @pre @p mat_a is not distributed
/// @pre @p mat_a has size (N x N)
/// @pre @p mat_a has blocksize (NB x NB)
/// @pre @p mat_a has tilesize (NB x NB)
///
/// @param mat_b contains the Cholesky factorisation of the Hermitian positive definite matrix B
/// The triangular matrix can be lower (L) or upper (U). Only the tiles of
/// the matrix which contain the lower triangular or the upper triangular part are accessed.
/// Note: B should be modifiable as the diagonal tiles might be temporarily modified during the calculation.
/// @pre @p mat_b is not distributed
/// @pre @p mat_b has size (N x N)
/// @pre @p mat_b has blocksize (NB x NB)
/// @pre @p mat_b has tilesize (NB x NB)
template <Backend backend, Device device, class T>
void generalized_to_standard(blas::Uplo uplo, Matrix<T, device>& mat_a, Matrix<T, device>& mat_b) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_size(mat_b), mat_b);
  DLAF_ASSERT(matrix::square_blocksize(mat_b), mat_b);
  DLAF_ASSERT(mat_a.size() == mat_b.size(), mat_a, mat_b);
  DLAF_ASSERT(mat_a.blockSize() == mat_b.blockSize(), mat_a, mat_b);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_a), mat_a);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_b), mat_b);
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);
  DLAF_ASSERT(matrix::local_matrix(mat_b), mat_b);

  switch (uplo) {
    case blas::Uplo::Lower:
      GenToStd<backend, device, T>::call_L(mat_a, mat_b);
      break;
    case blas::Uplo::Upper:
      GenToStd<backend, device, T>::call_U(mat_a, mat_b);
      break;
    case blas::Uplo::General:
      DLAF_UNIMPLEMENTED(uplo);
      break;
  }
}

/// Reduce a Hermitian definite generalized eigenproblem to standard form.
///
/// It computes inv(B)*A*inv(B**H) or inv(B**H)*A*inv(B), where B have been previously factorized
/// using the Cholesky factorization.
/// Implementation on distributed memory.
///
/// @param grid is the communicator grid on which the matrix A has been distributed,
/// @param uplo specifies if the elements of the Hermitian matrix A and the triangular matrix B
/// to be referenced are the elements in the lower or upper triangular part,
///
/// @param mat_a on entry it contains the Hermitian matrix A (if A is real, the matrix is symmetric),
/// on exit the matrix elements are overwritten with the elements of the matrix B.
/// Only the tiles of the matrix which contain the lower triangular or the upper triangular part are accessed.
/// @pre @p mat_a is distributed according to @p grid
/// @pre @p mat_a has size (N x N)
/// @pre @p mat_a has blocksize (NB x NB)
/// @pre @p mat_a has tilesize (NB x NB)
///
/// @param mat_b contains the triangular matrix. It can be lower (L) or upper (U). Only the tiles of
/// the matrix which contain the lower triangular or the upper triangular part are accessed.
/// Note: B should be modifiable as the diagonal tiles might be temporarily modified during the calculation.
/// @pre @p mat_b is distributed according to @p grid
/// @pre @p mat_b has size (N x N)
/// @pre @p mat_b has blocksize (NB x NB)
/// @pre @p mat_b has tilesize (NB x NB)
template <Backend backend, Device device, class T>
void generalized_to_standard(comm::CommunicatorGrid& grid, blas::Uplo uplo, Matrix<T, device>& mat_a,
                             Matrix<T, device>& mat_b) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_size(mat_b), mat_b);
  DLAF_ASSERT(matrix::square_blocksize(mat_b), mat_b);
  DLAF_ASSERT(mat_a.size() == mat_b.size(), mat_a, mat_b);
  DLAF_ASSERT(mat_a.blockSize() == mat_b.blockSize(), mat_a, mat_b);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_a), mat_a);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_b), mat_b);
  DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);
  DLAF_ASSERT(matrix::equal_process_grid(mat_b, grid), mat_b, grid);

  switch (uplo) {
    case blas::Uplo::Lower:
      GenToStd<backend, device, T>::call_L(grid, mat_a, mat_b);
      break;
    case blas::Uplo::Upper:
      GenToStd<backend, device, T>::call_U(grid, mat_a, mat_b);
      break;
    case blas::Uplo::General:
      DLAF_UNIMPLEMENTED(uplo);
      break;
  }
}

}
