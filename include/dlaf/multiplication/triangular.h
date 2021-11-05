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
#include "dlaf/matrix/matrix.h"
#include "dlaf/multiplication/triangular/api.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace multiplication {

/// Triangular Matrix multiplication implementation on local memory, computing B = alpha op(A)  B
/// (when side == Left) and B = alpha B op(A) (when side == Right)
///
/// @param side specifies whether op(A) appears on the \a Left or on the \a Right of matrix B,
/// @param uplo specifies whether the matrix A is a \a Lower or an \a Upper triangular matrix,
/// @param op specifies the form of op(A) to be used in the matrix multiplication: \a NoTrans, \a Trans,
/// \a ConjTrans,
/// @param diag specifies if the matrix A is assumed to be unit triangular (\a Unit) or not (\a NonUnit),
/// @param mat_a contains the triangular matrix A. Only the tiles of the matrix which contain the upper or
/// the lower triangular part (depending on the value of uplo) are accessed in read-only mode (the
/// elements are not modified),
/// @param mat_b on entry it contains the matrix B, on exit the matrix elements are overwritten with the
/// elements of the transformed matrix.
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
    DLAF_ASSERT(matrix::multipliable(mat_a, mat_b, mat_b, op, blas::Op::NoTrans), mat_a, mat_b, op);

    if (uplo == blas::Uplo::Lower) {
      if (op == blas::Op::NoTrans) {
        internal::Triangular<backend, device, T>::call_LLN(diag, alpha, mat_a, mat_b);
      }
      else {
        internal::Triangular<backend, device, T>::call_LLT(op, diag, alpha, mat_a, mat_b);
      }
    }
    else {
      if (op == blas::Op::NoTrans) {
        internal::Triangular<backend, device, T>::call_LUN(diag, alpha, mat_a, mat_b);
      }
      else {
        internal::Triangular<backend, device, T>::call_LUT(op, diag, alpha, mat_a, mat_b);
      }
    }
  }
  else {
    DLAF_ASSERT(matrix::multipliable(mat_b, mat_a, mat_b, blas::Op::NoTrans, op), mat_a, mat_b, op);

    if (uplo == blas::Uplo::Lower) {
      if (op == blas::Op::NoTrans) {
        internal::Triangular<backend, device, T>::call_RLN(diag, alpha, mat_a, mat_b);
      }
      else {
        internal::Triangular<backend, device, T>::call_RLT(op, diag, alpha, mat_a, mat_b);
      }
    }
    else {
      if (op == blas::Op::NoTrans) {
        internal::Triangular<backend, device, T>::call_RUN(diag, alpha, mat_a, mat_b);
      }
      else {
        internal::Triangular<backend, device, T>::call_RUT(op, diag, alpha, mat_a, mat_b);
      }
    }
  }
}

}
}
