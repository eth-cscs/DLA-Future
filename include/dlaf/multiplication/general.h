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

#include <blas.hh>

#include "dlaf/common/assert.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/multiplication/general/api.h"
#include "dlaf/util_matrix.h"

namespace dlaf::multiplication {

/// General sub-matrix multiplication implementation on local memory, computing
/// C[a:b][a:b] = alpha * opA(A[a:b][a:b]) * opB(B[a:b][a:b]) + beta * C[a:b][a:b]
/// where [a:b] is the range of tiles starting from tile index @p a to tile index @p b (included)
///
/// @param  opA specifies the form of opA(A) to be used in the matrix multiplication:
///         \a NoTrans, \a Trans, \a ConjTrans,
/// @param  opB specifies the form of opB(B) to be used in the matrix multiplication:
///         \a NoTrans, \a Trans, \a ConjTrans,
/// @param  mat_a contains the input matrix A. Only tiles whose both row and col tile coords are in
///         the closed range [a,b] are accessed in read-only mode (elements are not modified)
/// @param  mat_b contains the input matrix B. Only tiles whose both row and col tile coords are in
///         the closed range [a,b] are accessed in read-only mode (elements are not modified)
/// @param  mat_c On entry it contains the input matrix C. On exit matrix tiles in the range will be
///         overwritten with the result, while others are left untouched.
///         Only tiles whose both row and col tile coords are in the closed range [a,b] are accessed.
/// @pre mat_a, mat_b and mat_c have the same square block size,
/// @pre mat_a, mat_b and mat_c have the same square size,
/// @pre mat_a, mat_b and mat_c are not distributed,
/// @pre a <= b < mat_a.nrTiles().rows()
template <Backend B, Device D, class T>
void generalSubMatrix(const SizeType a, const SizeType b, const blas::Op opA, const blas::Op opB,
                      const T alpha, Matrix<const T, D>& mat_a, Matrix<const T, D>& mat_b, const T beta,
                      Matrix<T, D>& mat_c) {
  DLAF_ASSERT(dlaf::matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(dlaf::matrix::square_blocksize(mat_b), mat_b);
  DLAF_ASSERT(dlaf::matrix::square_blocksize(mat_c), mat_c);

  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);
  DLAF_ASSERT(matrix::local_matrix(mat_b), mat_b);
  DLAF_ASSERT(matrix::local_matrix(mat_c), mat_c);

  // Note:
  // This is an over-constraint, since the algorithm just cares about the sub-matrix size.
  // It simplifies next check about [a,b] range validity, that otherwise would require it to be
  // validated against every single sub-matrix in a, b and c that might have different element sizes.
  //
  // At the moment, we don't have this use-case, so let's keep it simple.
  DLAF_ASSERT(equal_size(mat_a, mat_b), mat_a, mat_b);
  DLAF_ASSERT(equal_size(mat_a, mat_c), mat_a, mat_c);

  const SizeType m = mat_a.nrTiles().rows();
  DLAF_ASSERT(a <= b, a, b);
  DLAF_ASSERT(a >= 0 && a < m, a, m);

  using namespace blas;

  if (opA == Op::NoTrans && opB == Op::NoTrans)
    internal::GeneralSub<B, D, T>::callNN(a, b, opA, opB, alpha, mat_a, mat_b, beta, mat_c);
  else
    DLAF_UNIMPLEMENTED(opA, opB);
}

template <Backend B, Device D, class T>
void generalSubMatrixK(comm::CommunicatorGrid grid, const SizeType a, const SizeType b, const T alpha,
                       Matrix<const T, D>& mat_a, Matrix<const T, D>& mat_b, const T beta,
                       Matrix<T, D>& mat_c) {
  DLAF_ASSERT(equal_process_grid(mat_a, grid), mat_a, grid);
  DLAF_ASSERT(equal_process_grid(mat_b, grid), mat_a, grid);
  DLAF_ASSERT(equal_process_grid(mat_c, grid), mat_a, grid);

  DLAF_ASSERT(dlaf::matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(dlaf::matrix::square_blocksize(mat_b), mat_b);
  DLAF_ASSERT(dlaf::matrix::square_blocksize(mat_c), mat_c);

  // Note:
  // This is an over-constraint, since the algorithm just cares about the sub-matrix size (and its
  // distribution).
  // It simplifies next check about [a,b] range validity, that otherwise would require it to be
  // validated against every single sub-matrix in a, b and c that might have different element sizes.
  //
  // At the moment, we don't have this use-case, so let's keep it simple.
  DLAF_ASSERT(equal_size(mat_a, mat_b), mat_a, mat_b);
  DLAF_ASSERT(equal_size(mat_a, mat_c), mat_a, mat_c);

  const SizeType m = mat_a.nrTiles().rows();
  DLAF_ASSERT(a <= b, a, b);
  DLAF_ASSERT(a >= 0 && a < m, a, m);

  internal::GeneralSubK<B, D, T>::callNN(grid, a, b, alpha, mat_a, mat_b, beta, mat_c);
}

}
