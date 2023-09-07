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

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/multiplication/triangular/api.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace dlaf {

/// Triangular Matrix multiplication implementation on local memory, computing B = alpha op(A)  B
/// (when side == Left) and B = alpha B op(A) (when side == Right)
///
/// @param side specifies whether op(A) appears on the \a Left or on the \a Right of matrix B,
/// @param uplo specifies whether the matrix A is a \a Lower or an \a Upper triangular matrix,
/// @param op specifies the form of op(A) to be used in the matrix multiplication: \a NoTrans, \a Trans,
/// \a ConjTrans,
/// @param diag specifies if the matrix A is assumed to be unit triangular (\a Unit) or not (\a NonUnit),
///
/// @param mat_a contains the triangular matrix A. Only the tiles of the matrix which contain the upper or
/// the lower triangular part (depending on the value of uplo) are accessed in read-only mode (the
/// elements are not modified),
/// @pre @p mat_a is not distributed
/// @pre @p mat_a has size (N x N)
/// @pre @p mat_a has blocksize (NB x NB)
/// @pre @p mat_a has tilesize (NB x NB)
///
/// @param mat_b on entry it contains the matrix B, on exit the matrix elements are overwritten with the
/// elements of the result.
/// @pre @p mat_b is not distributed
/// @pre @p mat_b has size (N x N)
/// @pre @p mat_b has blocksize (NB x NB)
/// @pre @p mat_b has tilesize (NB x NB)
template <Backend backend, Device device, class T>
void triangular_multiplication(blas::Side side, blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha,
                               Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_a), mat_a);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_b), mat_b);
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);
  DLAF_ASSERT(matrix::local_matrix(mat_b), mat_b);

  if (side == blas::Side::Left) {
    DLAF_ASSERT(matrix::multipliable(mat_a, mat_b, mat_b, op, blas::Op::NoTrans), mat_a, mat_b, op);

    if (uplo == blas::Uplo::Lower) {
      if (op == blas::Op::NoTrans) {
        multiplication::internal::Triangular<backend, device, T>::call_LLN(diag, alpha, mat_a, mat_b);
      }
      else {
        multiplication::internal::Triangular<backend, device, T>::call_LLT(op, diag, alpha, mat_a,
                                                                           mat_b);
      }
    }
    else {
      if (op == blas::Op::NoTrans) {
        multiplication::internal::Triangular<backend, device, T>::call_LUN(diag, alpha, mat_a, mat_b);
      }
      else {
        multiplication::internal::Triangular<backend, device, T>::call_LUT(op, diag, alpha, mat_a,
                                                                           mat_b);
      }
    }
  }
  else {
    DLAF_ASSERT(matrix::multipliable(mat_b, mat_a, mat_b, blas::Op::NoTrans, op), mat_a, mat_b, op);

    if (uplo == blas::Uplo::Lower) {
      if (op == blas::Op::NoTrans) {
        multiplication::internal::Triangular<backend, device, T>::call_RLN(diag, alpha, mat_a, mat_b);
      }
      else {
        multiplication::internal::Triangular<backend, device, T>::call_RLT(op, diag, alpha, mat_a,
                                                                           mat_b);
      }
    }
    else {
      if (op == blas::Op::NoTrans) {
        multiplication::internal::Triangular<backend, device, T>::call_RUN(diag, alpha, mat_a, mat_b);
      }
      else {
        multiplication::internal::Triangular<backend, device, T>::call_RUT(op, diag, alpha, mat_a,
                                                                           mat_b);
      }
    }
  }
}

/// Triangular Matrix multiplication implementation on distributed memory, computing B = alpha op(A)  B
/// (when side == Left) and B = alpha B op(A) (when side == Right)
///
/// @param side specifies whether op(A) appears on the \a Left or on the \a Right of matrix B,
/// @param uplo specifies whether the matrix A is a \a Lower or an \a Upper triangular matrix,
/// @param op specifies the form of op(A) to be used in the matrix multiplication: \a NoTrans, \a Trans,
/// \a ConjTrans,
/// @param diag specifies if the matrix A is assumed to be unit triangular (\a Unit) or not (\a NonUnit),
///
/// @param mat_a contains the triangular matrix A. Only the tiles of the matrix which contain the upper or
/// the lower triangular part (depending on the value of uplo) are accessed in read-only mode (the
/// elements are not modified),
/// @pre @p mat_a is distributed according to @p grid
/// @pre @p mat_a has size (N x N)
/// @pre @p mat_a has blocksize (NB x NB)
/// @pre @p mat_a has tilesize (NB x NB)
///
/// @param mat_b on entry it contains the matrix B, on exit the matrix elements are overwritten with the
/// elements of the result.
/// @pre @p mat_b is distributed according to @p grid
/// @pre @p mat_b has size (N x N)
/// @pre @p mat_b has blocksize (NB x NB)
/// @pre @p mat_b has tilesize (NB x NB)
template <Backend backend, Device device, class T>
void triangular_multiplication(comm::CommunicatorGrid grid, blas::Side side, blas::Uplo uplo,
                               blas::Op op, blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                               Matrix<T, device>& mat_b) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_a), mat_a);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_b), mat_b);
  DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);
  DLAF_ASSERT(matrix::equal_process_grid(mat_b, grid), mat_b, grid);

  if (side == blas::Side::Left) {
    DLAF_ASSERT(matrix::multipliable(mat_a, mat_b, mat_b, op, blas::Op::NoTrans), mat_a, mat_b, op);

    if (uplo == blas::Uplo::Lower) {
      if (op == blas::Op::NoTrans) {
        multiplication::internal::Triangular<backend, device, T>::call_LLN(grid, diag, alpha, mat_a,
                                                                           mat_b);
      }
      else {
        // Left Lower Trans/ConjTrans
        DLAF_UNIMPLEMENTED(side, uplo, op, diag);
      }
    }
    else {
      if (op == blas::Op::NoTrans) {
        multiplication::internal::Triangular<backend, device, T>::call_LUN(grid, diag, alpha, mat_a,
                                                                           mat_b);
      }
      else {
        // Left Upper Trans/ConjTrans
        DLAF_UNIMPLEMENTED(side, uplo, op, diag);
      }
    }
  }
  else {
    DLAF_ASSERT(matrix::multipliable(mat_b, mat_a, mat_b, blas::Op::NoTrans, op), mat_a, mat_b, op);

    if (uplo == blas::Uplo::Lower) {
      if (op == blas::Op::NoTrans) {
        multiplication::internal::Triangular<backend, device, T>::call_RLN(grid, diag, alpha, mat_a,
                                                                           mat_b);
      }
      else {
        // Right Lower Trans/ConjTrans
        DLAF_UNIMPLEMENTED(side, uplo, op, diag);
      }
    }
    else {
      if (op == blas::Op::NoTrans) {
        multiplication::internal::Triangular<backend, device, T>::call_RUN(grid, diag, alpha, mat_a,
                                                                           mat_b);
      }
      else {
        // Right Upper Trans/ConjTrans
        DLAF_UNIMPLEMENTED(side, uplo, op, diag);
      }
    }
  }
}
}
