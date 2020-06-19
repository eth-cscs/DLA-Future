//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <blas.hh>
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix.h"
#include "dlaf/solver/internal.h"
#include "dlaf/types.h"

namespace dlaf {

template <>
struct Solver<Backend::MC> {
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
  /// @pre matrix A has a square size,
  /// @pre matrix A has a square block size,
  /// @pre matrix A and matrix B are not distributed,
  /// @pre matrix A and matrix B are multipliable.
  template <class T>
  static void triangular(blas::Side side, blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha,
                         Matrix<const T, Device::CPU>& mat_a, Matrix<T, Device::CPU>& mat_b);

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
  template <class T>
  static void triangular(comm::CommunicatorGrid grid, blas::Side side, blas::Uplo uplo, blas::Op op,
                         blas::Diag diag, T alpha, Matrix<const T, Device::CPU>& mat_a,
                         Matrix<T, Device::CPU>& mat_b);
};

}

#include <dlaf/solver/triangular/mc.tpp>

/// ---- ETI
namespace dlaf {

#define DLAF_TRIANGULAR_ETI(KWORD, DATATYPE)                                                          \
  KWORD template void Solver<Backend::MC>::triangular<DATATYPE>(comm::CommunicatorGrid, blas::Side,   \
                                                                blas::Uplo, blas::Op op, blas::Diag,  \
                                                                DATATYPE,                             \
                                                                Matrix<const DATATYPE, Device::CPU>&, \
                                                                Matrix<DATATYPE, Device::CPU>&);      \
  KWORD template void Solver<Backend::MC>::triangular<DATATYPE>(blas::Side, blas::Uplo, blas::Op,     \
                                                                blas::Diag, DATATYPE,                 \
                                                                Matrix<const DATATYPE, Device::CPU>&, \
                                                                Matrix<DATATYPE, Device::CPU>&);

DLAF_TRIANGULAR_ETI(extern, float)
DLAF_TRIANGULAR_ETI(extern, double)
DLAF_TRIANGULAR_ETI(extern, std::complex<float>)
DLAF_TRIANGULAR_ETI(extern, std::complex<double>)

}
