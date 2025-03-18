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
void generalMatrix(comm::CommunicatorPipeline<comm::CommunicatorType::Row>& row_task_chain,
                   comm::CommunicatorPipeline<comm::CommunicatorType::Col>& col_task_chain,
                   const T alpha, MatrixRef<const T, D>& mat_a, MatrixRef<const T, D>& mat_b,
                   const T beta, MatrixRef<T, D>& mat_c) {
  DLAF_ASSERT(matrix::equal_process_grid(row_task_chain, col_task_chain), row_task_chain,
              col_task_chain);

  DLAF_ASSERT(matrix::equal_process_grid(mat_a, row_task_chain), mat_a, row_task_chain);
  DLAF_ASSERT(matrix::equal_process_grid(mat_b, row_task_chain), mat_b, row_task_chain);
  DLAF_ASSERT(matrix::equal_process_grid(mat_c, row_task_chain), mat_c, row_task_chain);

  DLAF_ASSERT_HEAVY(matrix::multipliable(mat_a, mat_b, mat_c, blas::Op::NoTrans, blas::Op::NoTrans),
                    mat_a, mat_b, mat_c);

  internal::General<B, D, T>::callNN(row_task_chain, col_task_chain, alpha, mat_a, mat_b, beta, mat_c);
}
}
