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
#include <dlaf/matrix/matrix.h>
#include <dlaf/multiplication/hermitian/api.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace dlaf {

/// Hermitian Matrix multiplication implementation on local memory, computing C = beta C + alpha A B
/// (when side == Left) or C + alpha B A (when side == Right), where A is a Hermitian matrix.
///
/// @param side specifies whether A appears on the \a Left or on the \a Right of matrix B,
/// @param uplo specifies if the elements of the Hermitian matrix A to be referenced are the elements in
/// the lower or upper triangular part,
///
/// @param mat_a contains the hermitian matrix A. Only the tiles of the matrix which contain the upper or
/// the lower triangular part which represent the Hermitian matrix (depending on the value of uplo)
/// are accessed in read-only mode (the elements are not modified),
/// @pre @p mat_b is not distributed
/// @pre @p mat_b has size (N x M)
/// @pre @p mat_b has blocksize (NB x NB)
/// @pre @p mat_b has tilesize (NB x NB)
///
/// @param mat_b contains the matrix B accessed in read-only mode (the elements are not modified),
/// @pre @p mat_b is not distributed
/// @pre @p mat_b has size (M x K)
/// @pre @p mat_b has blocksize (NB x NB)
/// @pre @p mat_b has tilesize (NB x NB)
///
/// @param mat_c on entry it contains the matrix C, on exit the matrix elements are overwritten with the
/// elements of the result.
/// @pre @p mat_b is not distributed
/// @pre @p mat_b has size (N x K)
/// @pre @p mat_b has blocksize (NB x NB)
/// @pre @p mat_b has tilesize (NB x NB)
template <Backend B, Device D, class T>
void hermitian_multiplication(blas::Side side, blas::Uplo uplo, const T alpha, Matrix<const T, D>& mat_a,
                              Matrix<const T, D>& mat_b, const T beta, Matrix<T, D>& mat_c) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_a), mat_a);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_b), mat_b);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_c), mat_c);
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);
  DLAF_ASSERT(matrix::local_matrix(mat_b), mat_b);
  DLAF_ASSERT(matrix::local_matrix(mat_c), mat_c);

  if (side == blas::Side::Left) {
    DLAF_ASSERT(matrix::multipliable(mat_a, mat_b, mat_c, blas::Op::NoTrans, blas::Op::NoTrans), mat_a,
                mat_b, mat_c);
    switch (uplo) {
      case blas::Uplo::Lower:
        return multiplication::internal::Hermitian<B, D, T>::call_LL(alpha, mat_a, mat_b, beta, mat_c);
      case blas::Uplo::Upper:
        DLAF_UNIMPLEMENTED(uplo);
        break;
      case blas::Uplo::General:
        DLAF_UNIMPLEMENTED(uplo);
        break;
    }
  }
  else {
    DLAF_ASSERT(matrix::multipliable(mat_b, mat_a, mat_c, blas::Op::NoTrans, blas::Op::NoTrans), mat_a,
                mat_b, mat_c);
    DLAF_UNIMPLEMENTED(side);
  }
}

/// Hermitian Matrix multiplication implementation on distributed memory, computing
/// C = beta C + alpha A B (when side == Left) or C + alpha B A (when side == Right),
/// where A is a Hermitian matrix.
///
/// @param side specifies whether A appears on the \a Left or on the \a Right of matrix B,
/// @param uplo specifies if the elements of the Hermitian matrix A to be referenced are the elements in
/// the lower or upper triangular part,
///
/// @param mat_a contains the hermitian matrix A. Only the tiles of the matrix which contain the upper or
/// the lower triangular part which represent the Hermitian matrix (depending on the value of uplo)
/// are accessed in read-only mode (the elements are not modified),
/// @pre @p mat_b is distributed according to @p grid
/// @pre @p mat_b has size (N x M)
/// @pre @p mat_b has blocksize (NB x NB)
/// @pre @p mat_b has tilesize (NB x NB)
///
/// @param mat_b contains the matrix B accessed in read-only mode (the elements are not modified),
/// @pre @p mat_b is distributed according to @p grid
/// @pre @p mat_b has size (M x K)
/// @pre @p mat_b has blocksize (NB x NB)
/// @pre @p mat_b has tilesize (NB x NB)
///
/// @param mat_c on entry it contains the matrix C, on exit the matrix elements are overwritten with the
/// elements of the result.
/// @pre @p mat_b is distributed according to @p grid
/// @pre @p mat_b has size (N x K)
/// @pre @p mat_b has blocksize (NB x NB)
/// @pre @p mat_b has tilesize (NB x NB)
template <Backend B, Device D, class T>
void hermitian_multiplication(comm::CommunicatorGrid& grid, blas::Side side, blas::Uplo uplo,
                              const T alpha, Matrix<const T, D>& mat_a, Matrix<const T, D>& mat_b,
                              const T beta, Matrix<T, D>& mat_c) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_a), mat_a);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_b), mat_b);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_c), mat_c);
  DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);
  DLAF_ASSERT(matrix::equal_process_grid(mat_b, grid), mat_b, grid);
  DLAF_ASSERT(matrix::equal_process_grid(mat_c, grid), mat_c, grid);

  if (side == blas::Side::Left) {
    DLAF_ASSERT(matrix::multipliable(mat_a, mat_b, mat_c, blas::Op::NoTrans, blas::Op::NoTrans), mat_a,
                mat_b, mat_c);
    switch (uplo) {
      case blas::Uplo::Lower:
        return multiplication::internal::Hermitian<B, D, T>::call_LL(grid, alpha, mat_a, mat_b, beta,
                                                                     mat_c);
      case blas::Uplo::Upper:
        DLAF_UNIMPLEMENTED(uplo);
        break;
      case blas::Uplo::General:
        DLAF_UNIMPLEMENTED(uplo);
        break;
    }
  }
  else {
    DLAF_ASSERT(matrix::multipliable(mat_b, mat_a, mat_c, blas::Op::NoTrans, blas::Op::NoTrans), mat_a,
                mat_b, mat_c);
    DLAF_UNIMPLEMENTED(side);
  }
}

}
