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

#include <blas.hh>

#include <dlaf/common/assert.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_ref.h>
#include <dlaf/multiplication/general/api.h>
#include <dlaf/util_matrix.h>

namespace dlaf::multiplication::internal {

/// General matrix multiplication implementation on local memory, computing
/// C = alpha * opA(A) * opB(B) + beta * C
///
/// @param  opA specifies the form of opA(A) to be used in the matrix multiplication:
///         \a currently only NoTrans,
/// @param  opB specifies the form of opB(B) to be used in the matrix multiplication:
///         \a currently only NoTrans,
///
/// @param  mat_a contains the input matrix A.
/// @pre @p mat_a is not distributed
///
/// @param  mat_b contains the input matrix B.
/// @pre @p mat_b is not distributed
///
/// @param  mat_c On entry it contains the input matrix C. On exit matrix will be overwritten with the
///         result, while others are left untouched.
/// @pre @p mat_c is not distributed
///
/// @pre multipliable_sizes(mat_a.size(), mat_b.size(), mat_c.size(), opA, opB)
/// @pre multipliable_sizes(mat_a.tile_size(), mat_b.tile_size(), mat_c.tile_size(), opA, opB)
/// @pre multipliable_sizes(mat_a.tile_size_of({0, 0}), mat_b.tile_size_of({0, 0}),
///      mat_c.tile_size_of({0, 0}), opA, opB)
template <Backend B, Device D, class T>
void generalMatrix(const blas::Op opA, const blas::Op opB, const T alpha, MatrixRef<const T, D>& mat_a,
                   MatrixRef<const T, D>& mat_b, const T beta, MatrixRef<T, D>& mat_c) {
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);
  DLAF_ASSERT(matrix::local_matrix(mat_b), mat_b);
  DLAF_ASSERT(matrix::local_matrix(mat_c), mat_c);

  DLAF_ASSERT_HEAVY(matrix::multipliable(mat_a, mat_b, mat_c, blas::Op::NoTrans, blas::Op::NoTrans),
                    mat_a, mat_b, mat_c);

  if (opA == blas::Op::NoTrans && opB == blas::Op::NoTrans)
    internal::General<B, D, T>::callNN(alpha, mat_a, mat_b, beta, mat_c);
  else
    DLAF_UNIMPLEMENTED(opA, opB);
}

/// General sub-matrix distributed multiplication, computing
/// C = alpha * A * B + beta * C
///
/// @param  mat_a contains the input matrix A.
/// @param  mat_b contains the input matrix B.
/// @param  mat_c On entry it contains the input matrix C. On exit matrix tiles in the range will be
///         overwritten with the result, while others are left untouched.
///
/// @pre @p mat_a, @p mat_b and @p mat_c are distributed on the same grid,
/// @pre multipliable_sizes(mat_a.size(), mat_b.size(), mat_c.size(), opA, opB)
/// @pre multipliable_sizes(mat_a.tile_size(), mat_b.tile_size(), mat_c.tile_size(), opA, opB)
/// @pre multipliable_sizes(mat_a.tile_size_of({0, 0}), mat_b.tile_size_of({0, 0}),
///      mat_c.tile_size_of({0, 0}), opA, opB)
template <Backend B, Device D, class T>
void generalMatrix(common::Pipeline<comm::Communicator>& row_task_chain,
                   common::Pipeline<comm::Communicator>& col_task_chain, const T alpha,
                   MatrixRef<const T, D>& mat_a, MatrixRef<const T, D>& mat_b, const T beta,
                   MatrixRef<T, D>& mat_c) {
  DLAF_ASSERT(matrix::same_process_grid(mat_c, mat_a), mat_c, mat_b);
  DLAF_ASSERT(matrix::same_process_grid(mat_c, mat_b), mat_c, mat_b);

  DLAF_ASSERT_HEAVY(matrix::multipliable(mat_a, mat_b, mat_c, blas::Op::NoTrans, blas::Op::NoTrans),
                    mat_a, mat_b, mat_c);

  internal::General<B, D, T>::callNN(row_task_chain, col_task_chain, alpha, mat_a, mat_b, beta, mat_c);
}

/// General sub-matrix multiplication implementation on local memory, computing
/// C[a:b][a:b] = alpha * opA(A[a:b][a:b]) * opB(B[a:b][a:b]) + beta * C[a:b][a:b]
/// where [a:b] is the range of tiles starting from tile index @p a to tile index @p b (excluded)
///
/// @param  opA specifies the form of opA(A) to be used in the matrix multiplication:
///         \a NoTrans, \a Trans, \a ConjTrans,
/// @param  opB specifies the form of opB(B) to be used in the matrix multiplication:
///         \a NoTrans, \a Trans, \a ConjTrans,
///
/// @param  mat_a contains the input matrix A. Only tiles whose both row and col tile coords are in
///         the closed range [a,b] are accessed in read-only mode (elements are not modified)
/// @pre @p mat_a is not distributed
/// @pre @p mat_a has size (N x N)
/// @pre @p mat_a has blocksize (NB x NB)
/// @pre @p mat_a has tilesize (NB x NB)
///
/// @param  mat_b contains the input matrix B. Only tiles whose both row and col tile coords are in
///         the closed range [a,b] are accessed in read-only mode (elements are not modified)
/// @pre @p mat_b is not distributed
/// @pre @p mat_b has size (N x N)
/// @pre @p mat_b has blocksize (NB x NB)
/// @pre @p mat_b has tilesize (NB x NB)
///
/// @param  mat_c On entry it contains the input matrix C. On exit matrix tiles in the range will be
///         overwritten with the result, while others are left untouched.
///         Only tiles whose both row and col tile coords are in the closed range [a,b] are accessed.
/// @pre @p mat_c is not distributed
/// @pre @p mat_c has size (N x N)
/// @pre @p mat_c has blocksize (NB x NB)
/// @pre @p mat_c has tilesize (NB x NB)
///
/// @pre `a <= b <= mat_a.nrTiles().rows()`
template <Backend B, Device D, class T>
void generalSubMatrix(const SizeType a, const SizeType b, const blas::Op opA, const blas::Op opB,
                      const T alpha, Matrix<const T, D>& mat_a, Matrix<const T, D>& mat_b, const T beta,
                      Matrix<T, D>& mat_c) {
  DLAF_ASSERT(dlaf::matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(dlaf::matrix::square_blocksize(mat_b), mat_b);
  DLAF_ASSERT(dlaf::matrix::square_blocksize(mat_c), mat_c);

  DLAF_ASSERT(matrix::single_tile_per_block(mat_a), mat_a);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_b), mat_b);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_c), mat_c);

  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);
  DLAF_ASSERT(matrix::local_matrix(mat_b), mat_b);
  DLAF_ASSERT(matrix::local_matrix(mat_c), mat_c);

  // Note:
  // This is an over-constraint, since the algorithm just cares about the sub-matrix size.
  // It simplifies next check about [a,b) range validity, that otherwise would require it to be
  // validated against every single sub-matrix in a, b and c that might have different element sizes.
  //
  // At the moment, we don't have this use-case, so let's keep it simple.
  DLAF_ASSERT(dlaf::matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(equal_size(mat_a, mat_b), mat_a, mat_b);
  DLAF_ASSERT(equal_size(mat_a, mat_c), mat_a, mat_c);

  [[maybe_unused]] const SizeType m = mat_a.nrTiles().rows();
  DLAF_ASSERT(a <= b, a, b);
  DLAF_ASSERT(a >= 0 && a <= m, a, m);
  DLAF_ASSERT(b >= 0 && b <= m, b, m);

  using namespace blas;

  if (opA == Op::NoTrans && opB == Op::NoTrans)
    internal::GeneralSub<B, D, T>::callNN(a, b, opA, opB, alpha, mat_a, mat_b, beta, mat_c);
  else
    DLAF_UNIMPLEMENTED(opA, opB);
}

/// General sub-matrix distributed multiplication, computing
/// C[a:b][a:b] = alpha * A[a:b][a:b] * B[a:b][a:b] + beta * C[a:b][a:b]
/// where [a:b] is the range of tiles starting from tile index @p a to tile index @p b (excluded)
///
/// @param  mat_a contains the input matrix A. Only tiles whose both row and col tile coords are in
///         the closed range [a,b] are accessed in read-only mode (elements are not modified)
/// @pre @p mat_a is distributed according to @p grid
/// @pre @p mat_a has size (N x N)
/// @pre @p mat_a has blocksize (NB x NB)
/// @pre @p mat_a has tilesize (NB x NB)
///
/// @param  mat_b contains the input matrix B. Only tiles whose both row and col tile coords are in
///         the closed range [a,b] are accessed in read-only mode (elements are not modified)
/// @pre @p mat_b is distributed according to @p grid
/// @pre @p mat_b has size (N x N)
/// @pre @p mat_b has blocksize (NB x NB)
/// @pre @p mat_b has tilesize (NB x NB)
///
/// @param  mat_c On entry it contains the input matrix C. On exit matrix tiles in the range will be
///         overwritten with the result, while others are left untouched.
///         Only tiles whose both row and col tile coords are in the closed range [a,b] are accessed.
/// @pre @p mat_c is distributed according to @p grid
/// @pre @p mat_c has size (N x N)
/// @pre @p mat_c has blocksize (NB x NB)
/// @pre @p mat_c has tilesize (NB x NB)
///
/// @pre `a <= b <= mat_a.nrTiles().rows()`
template <Backend B, Device D, class T>
void generalSubMatrix(comm::CommunicatorPipeline<comm::CommunicatorType::Row>& row_task_chain,
                      comm::CommunicatorPipeline<comm::CommunicatorType::Col>& col_task_chain,
                      const SizeType a, const SizeType b, const T alpha, Matrix<const T, D>& mat_a,
                      Matrix<const T, D>& mat_b, const T beta, Matrix<T, D>& mat_c) {
  DLAF_ASSERT(equal_process_grid(mat_a, row_task_chain), mat_a, row_task_chain);
  DLAF_ASSERT(equal_process_grid(mat_b, row_task_chain), mat_a, row_task_chain);
  DLAF_ASSERT(equal_process_grid(mat_c, row_task_chain), mat_a, row_task_chain);

  DLAF_ASSERT(equal_process_grid(mat_a, col_task_chain), mat_a, col_task_chain);
  DLAF_ASSERT(equal_process_grid(mat_b, col_task_chain), mat_a, col_task_chain);
  DLAF_ASSERT(equal_process_grid(mat_c, col_task_chain), mat_a, col_task_chain);

  DLAF_ASSERT(dlaf::matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(dlaf::matrix::square_blocksize(mat_b), mat_b);
  DLAF_ASSERT(dlaf::matrix::square_blocksize(mat_c), mat_c);

  DLAF_ASSERT(matrix::single_tile_per_block(mat_a), mat_a);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_b), mat_b);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_c), mat_c);

  // Note:
  // This is an over-constraint, since the algorithm just cares about the sub-matrix size (and its
  // distribution).
  // It simplifies next check about [a,b) range validity, that otherwise would require it to be
  // validated against every single sub-matrix in a, b and c that might have different element sizes.
  //
  // At the moment, we don't have this use-case, so let's keep it simple.
  DLAF_ASSERT(dlaf::matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(equal_size(mat_a, mat_b), mat_a, mat_b);
  DLAF_ASSERT(equal_size(mat_a, mat_c), mat_a, mat_c);

  [[maybe_unused]] const SizeType m = mat_a.nrTiles().rows();
  DLAF_ASSERT(a <= b, a, b);
  DLAF_ASSERT(a >= 0 && a <= m, a, m);
  DLAF_ASSERT(b >= 0 && b <= m, b, m);

  internal::GeneralSub<B, D, T>::callNN(row_task_chain, col_task_chain, a, b, alpha, mat_a, mat_b, beta,
                                        mat_c);
}

template <Backend B, Device D, class T>
void generalSubMatrix(comm::CommunicatorGrid& grid, const SizeType a, const SizeType b, const T alpha,
                      Matrix<const T, D>& mat_a, Matrix<const T, D>& mat_b, const T beta,
                      Matrix<T, D>& mat_c) {
  auto row_task_chain = grid.row_communicator_pipeline();
  auto col_task_chain = grid.col_communicator_pipeline();
  generalSubMatrix<B, D, T>(row_task_chain, col_task_chain, a, b, alpha, mat_a, mat_b, beta, mat_c);
}

}
