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

#include <blas.hh>
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/eigensolver/gen_to_std/impl.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {

/// Reduce a Hermitian definite generalized eigenproblem to standard form.
///
/// It computes inv(B)*A*inv(B**H) or inv(B**H)*A*inv(B), where B have been previously factorized
/// using the Cholesky factorization.
/// Implementation on local memory.
///
/// @param uplo specifies if the elements of the Hermitian matrix A and the triangular matrix B
/// to be referenced are the elements in the lower or upper triangular part,
/// @param mat_a on entry it contains the Hermitian matrix A (if A is real, the matrix is symmetric),
/// on exit the matrix elements are overwritten with the elements of the matrix B.
/// Only the tiles of the matrix which contain the lower triangular or the upper triangular part are accessed.
/// @param mat_b contains the triangular matrix. It can be lower (L) or upper (U). Only the tiles of
/// the matrix which contain the lower triangular or the upper triangular part are accessed.
/// Note: B should be modifiable as the diagonal tiles might be temporarly modified during the calculation.
/// @pre mat_a and mat_b have the same square size,
/// @pre mat_a and mat_b have the same square block size,
/// @pre mat_a and mat_b are not distributed.
template <Backend backend, Device device, class T>
void genToStd(blas::Uplo uplo, Matrix<T, device>& mat_a, Matrix<T, device>& mat_b) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_size(mat_b), mat_b);
  DLAF_ASSERT(matrix::square_blocksize(mat_b), mat_b);
  DLAF_ASSERT(mat_a.size() == mat_b.size(), mat_a, mat_b);
  DLAF_ASSERT(mat_a.blockSize() == mat_b.blockSize(), mat_a, mat_b);
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);
  DLAF_ASSERT(matrix::local_matrix(mat_b), mat_b);

  switch (uplo) {
    case blas::Uplo::Lower:
      internal::GenToStd<backend, device, T>::call_L(mat_a, mat_b);
      break;
    case blas::Uplo::Upper:
      DLAF_UNIMPLEMENTED(uplo);
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
/// @param mat_a on entry it contains the Hermitian matrix A (if A is real, the matrix is symmetric),
/// on exit the matrix elements are overwritten with the elements of the matrix B.
/// Only the tiles of the matrix which contain the lower triangular or the upper triangular part are accessed.
/// @param mat_b contains the triangular matrix. It can be lower (L) or upper (U). Only the tiles of
/// the matrix which contain the lower triangular or the upper triangular part are accessed.
/// Note: B should be modifiable as the diagonal tiles might be temporarly modified during the calculation.
/// @pre mat_a and mat_b have the same square size,
/// @pre mat_a and mat_b have the same square block size,
/// @pre mat_a and mat_b are distributed according to the grid.
template <Backend backend, Device device, class T>
void genToStd(comm::CommunicatorGrid grid, blas::Uplo uplo, Matrix<T, device>& mat_a,
              Matrix<T, device>& mat_b) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_size(mat_b), mat_b);
  DLAF_ASSERT(matrix::square_blocksize(mat_b), mat_b);
  DLAF_ASSERT(mat_a.size() == mat_b.size(), mat_a, mat_b);
  DLAF_ASSERT(mat_a.blockSize() == mat_b.blockSize(), mat_a, mat_b);
  DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);
  DLAF_ASSERT(matrix::equal_process_grid(mat_b, grid), mat_b, grid);

  switch (uplo) {
    case blas::Uplo::Lower:
      internal::GenToStd<backend, device, T>::call_L(grid, mat_a, mat_b);
      break;
    case blas::Uplo::Upper:
      DLAF_UNIMPLEMENTED(uplo);
      break;
    case blas::Uplo::General:
      DLAF_UNIMPLEMENTED(uplo);
      break;
  }
}

}
}
