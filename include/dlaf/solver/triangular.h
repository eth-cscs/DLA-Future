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
#include "dlaf/communication/mech.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/solver/triangular/mc.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace solver {

/// Triangular Solve implementation on local memory, solving op(A) X = alpha B (when side == Left)
/// or X op(A) = alpha B (when side == Right).
///
/// @param side specifies whether op(A) appears on the \a Left or on the \a Right of matrix X,
/// @param uplo specifies whether the matrix A is a \a Lower or \a Upper triangular matrix,
/// @param op specifies the form of op(A) to be used in the matrix multiplication: \a NoTrans, \a Trans,
/// \a ConjTrans,
/// @param diag specifies if the matrix A is assumed to be unit triangular (\a Unit) or not (\a NonUnit),
/// @param mat_a contains the triangular matrix A. Only the tiles of the matrix which contain the upper or
/// the lower triangular part (depending on the value of uplo) are accessed in read-only mode (the
/// elements are not modified),
/// @param mat_b on entry it contains the matrix B, on exit the matrix elements are overwritten with the
/// elements of the matrix X,
/// @pre mat_a has a square size,
/// @pre mat_a has a square block size,
/// @pre mat_a and mat_b are not distributed,
/// @pre mat_a and mat_b are multipliable.
template <Backend backend, Device device, class T>
void triangular(blas::Side side, blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha,
                Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);
  DLAF_ASSERT(matrix::local_matrix(mat_b), mat_b);

  if (side == blas::Side::Left) {
    // Check if A and B dimensions are compatible
    DLAF_ASSERT(matrix::multipliable(mat_a, mat_b, mat_b, op, blas::Op::NoTrans), mat_a, mat_b, op);

    if (uplo == blas::Uplo::Lower) {
      if (op == blas::Op::NoTrans) {
        // Left Lower NoTrans
        internal::Triangular<backend, device, T>::call_LLN(diag, alpha, mat_a, mat_b);
      }
      else {
        // Left Lower Trans/ConjTrans
        internal::Triangular<backend, device, T>::call_LLT(op, diag, alpha, mat_a, mat_b);
      }
    }
    else {
      if (op == blas::Op::NoTrans) {
        // Left Upper NoTrans
        internal::Triangular<backend, device, T>::call_LUN(diag, alpha, mat_a, mat_b);
      }
      else {
        // Left Upper Trans/ConjTrans
        internal::Triangular<backend, device, T>::call_LUT(op, diag, alpha, mat_a, mat_b);
      }
    }
  }
  else {
    // Check if A and B dimensions are compatible
    DLAF_ASSERT(matrix::multipliable(mat_b, mat_a, mat_b, blas::Op::NoTrans, op), mat_a, mat_b, op);

    if (uplo == blas::Uplo::Lower) {
      if (op == blas::Op::NoTrans) {
        // Right Lower NoTrans
        internal::Triangular<backend, device, T>::call_RLN(diag, alpha, mat_a, mat_b);
      }
      else {
        // Right Lower Trans/ConjTrans
        internal::Triangular<backend, device, T>::call_RLT(op, diag, alpha, mat_a, mat_b);
      }
    }
    else {
      if (op == blas::Op::NoTrans) {
        // Right Upper NoTrans
        internal::Triangular<backend, device, T>::call_RUN(diag, alpha, mat_a, mat_b);
      }
      else {
        // Right Upper Trans/ConjTrans
        internal::Triangular<backend, device, T>::call_RUT(op, diag, alpha, mat_a, mat_b);
      }
    }
  }
}

/// Triangular Solve implementation on distributed memory, solving op(A) X = alpha B (when side ==
/// Left) or X op(A) = alpha B (when side == Right).
///  Algorithm 1: matrix A is communicated.
///
/// @param side specifies whether op(A) appears on the \a Left or on the \a Right of matrix X,
/// @param uplo specifies whether the matrix A is a \a Lower or \a Upper triangular matrix,
/// @param op specifies the form of op(A) to be used in the matrix multiplication: \a NoTrans, \a
/// Trans, \a ConjTrans,
/// @param diag specifies if the matrix A is assumed to be unit triangular (\a Unit) or not (\a
/// NonUnit),
/// @param mat_a contains the triangular matrix A. Only the tiles of the matrix which contain the upper
/// or the lower triangular part (depending on the value of uplo) are accessed in read-only mode (the
/// elements are not modified),
/// @param mat_b on entry it contains the matrix B, on exit the matrix elements are overwritten with
/// the elements of the matrix X,
/// @pre matrix A has a square size,
/// @pre matrix A has a square block size,
/// @pre matrix A and matrix B are distributed according to the grid,
/// @pre matrix A and matrix B are multipliable.
template <Backend backend, Device device, class T>
void triangular(comm::CommunicatorGrid grid, blas::Side side, blas::Uplo uplo, blas::Op op,
                blas::Diag diag, T alpha, Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b,
                comm::MPIMech mech = comm::MPIMech::Yielding) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);
  DLAF_ASSERT(matrix::equal_process_grid(mat_b, grid), mat_b, grid);

  if (side == blas::Side::Left) {
    // Check if A and B dimensions are compatible
    DLAF_ASSERT(matrix::multipliable(mat_a, mat_b, mat_b, op, blas::Op::NoTrans), mat_a, mat_b, op);

    if (uplo == blas::Uplo::Lower) {
      if (op == blas::Op::NoTrans) {
        // Left Lower NoTrans
        internal::Triangular<backend, device, T>::call_LLN(grid, diag, alpha, mat_a, mat_b, mech);
      }
      else {
        // Left Lower Trans/ConjTrans
        DLAF_UNIMPLEMENTED(side, uplo, op, diag);
      }
    }
    else {
      if (op == blas::Op::NoTrans) {
        // Left Upper NoTrans
        DLAF_UNIMPLEMENTED(side, uplo, op, diag);
      }
      else {
        // Left Upper Trans/ConjTrans
        DLAF_UNIMPLEMENTED(side, uplo, op, diag);
      }
    }
  }
  else {
    // Check if A and B dimensions are compatible
    DLAF_ASSERT(matrix::multipliable(mat_a, mat_b, mat_b, blas::Op::NoTrans, op), mat_a, mat_b, op);

    if (uplo == blas::Uplo::Lower) {
      if (op == blas::Op::NoTrans) {
        // Right Lower NoTrans
        DLAF_UNIMPLEMENTED(side, uplo, op, diag);
      }
      else {
        // Right Lower Trans/ConjTrans
        DLAF_UNIMPLEMENTED(side, uplo, op, diag);
      }
    }
    else {
      if (op == blas::Op::NoTrans) {
        // Right Upper NoTrans
        DLAF_UNIMPLEMENTED(side, uplo, op, diag);
      }
      else {
        // Right Upper Trans/ConjTrans
        DLAF_UNIMPLEMENTED(side, uplo, op, diag);
      }
    }
  }
}

}
}
