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

/// Reduce a Hermitian definite generalized eigenproblem to standard form, using the factorization
/// obtained from potrf (Cholesky factorization): A <- f(A, B).
/// It solves B=inv(L)*A*inv(L**H) or B=inv(U**H)*A*inv(U).
/// Implementation on local memory.
///
/// @param mat_a on entry it contains the Hermitian matrix A (if A is real, the matrix is symmetric),
/// on exit the matrix elements are overwritten with the elements of the matrix B.
/// @param mat_b contains the triangular matrix. It can be lower (L) or upper (U). Only the tiles of
/// the matrix which contain the lower triangular or the upper triangular part are accessed.
/// @pre mat_a and mat_b have a square size,
/// @pre mat_a and mat_b have a square block size,
/// @pre mat_a and mat_b are not distributed.
template <Backend backend, Device device, class T>
void genToStd(blas::Uplo uplo, Matrix<T, device>& mat_a, Matrix<T, device>& mat_b) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_size(mat_b), mat_b);
  DLAF_ASSERT(matrix::square_blocksize(mat_b), mat_b);
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

template <Backend backend, Device device, class T>
void genToStd(comm::CommunicatorGrid grid, blas::Uplo uplo, Matrix<T, device>& mat_a,
              Matrix<T, device>& mat_b) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_size(mat_b), mat_b);
  DLAF_ASSERT(matrix::square_blocksize(mat_b), mat_b);
  DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);
  DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);

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
