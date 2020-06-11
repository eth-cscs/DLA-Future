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
#include "dlaf/factorization/internal.h"
#include "dlaf/matrix.h"
#include "dlaf/types.h"

namespace dlaf {

template <>
struct Factorization<Backend::MC> {
  /// Cholesky factorization which computes the factorization of an Hermitian positive
  /// definite matrix A.
  ///
  /// The factorization has the form A=LL^H (uplo = Lower) or A = U^H U (uplo = Upper),
  /// where L is a lower and U is an upper triangular matrix
  /// @param uplo specifies if the elements of the Hermitian matrix to be referenced are the elements in
  /// the lower or upper triangular part.
  /// @param mat_a on entry it contains the triangular matrix A, on exit the matrix elements
  /// are overwritten with the elements of the Cholesky factor. Only the tiles of the matrix
  /// which contain the upper or the lower triangular part (depending on the value of uplo).
  /// @pre mat_a has a square size
  /// @pre mat_a has a square block size
  /// @pre mat_a is not distributed
  template <class T>
  static void cholesky(blas::Uplo uplo, Matrix<T, Device::CPU>& mat_a);

  /// Cholesky factorization which computes the factorization of an Hermitian positive
  /// definite matrix A.
  ///
  /// The factorization has the form A=LL^H (uplo = Lower) or A = U^H U (uplo = Upper),
  /// where L is a lower and U is an upper triangular matrix
  /// @param grid is the communicator grid on which the matrix A has been distributed.
  /// @param uplo specifies if the elements of the Hermitian matrix to be referenced are the elements in
  /// the lower or upper triangular part.
  /// @param mat_a on entry it contains the triangular matrix A, on exit the matrix elements
  /// are overwritten with the elements of the Cholesky factor. Only the tiles of the matrix
  /// which contain the upper or the lower triangular part (depending on the value of uplo).
  /// @pre mat_a has a square size
  /// @pre mat_a has a square block size
  /// @pre mat_a is distributed according to grid.
  template <class T>
  static void cholesky(comm::CommunicatorGrid grid, blas::Uplo uplo, Matrix<T, Device::CPU>& mat_a);
};

}

#include <dlaf/factorization/cholesky/mc.tpp>

/// ---- ETI
namespace dlaf {

#define DLAF_CHOLESKY_ETI(KWORD, DATATYPE)                                                            \
  KWORD template void Factorization<Backend::MC>::cholesky<DATATYPE>(comm::CommunicatorGrid,          \
                                                                     blas::Uplo,                      \
                                                                     Matrix<DATATYPE, Device::CPU>&); \
  KWORD template void Factorization<Backend::MC>::cholesky<DATATYPE>(blas::Uplo,                      \
                                                                     Matrix<DATATYPE, Device::CPU>&);

DLAF_CHOLESKY_ETI(extern, float)
DLAF_CHOLESKY_ETI(extern, double)
DLAF_CHOLESKY_ETI(extern, std::complex<float>)
DLAF_CHOLESKY_ETI(extern, std::complex<double>)

}
