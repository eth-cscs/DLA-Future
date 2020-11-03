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
#include "dlaf/eigensolver/internal.h"
#include "dlaf/matrix.h"
#include "dlaf/types.h"

namespace dlaf {

template <>
struct Eigensolver<Backend::MC> {
  /// Reduce a Hermitian definite generalized eigenproblem to standard form, using the factorization
  /// obtained from potrf (Cholesky factorization): A <- f(A, B).
  /// It solves B=inv(L)*A*inv(L**H) or B=inv(U**H)*A*inv(U).
  /// Implementation on local memory.
  ///
  /// @param mat_a on entry it contains the Hermitian matrix A, on exit the matrix elements
  /// are overwritten with the elements of the matrix B.
  /// @param mat_b contains the triangular matrix. It can be lower (L) or upper (U). Only the tiles of
  /// the matrix which contain the lower triangular or the upper triangular part are accessed.
  template <class T>
  static void genToStd(blas::Uplo uplo, Matrix<T, Device::CPU>& mat_a, Matrix<T, Device::CPU>& mat_b);
};

}

#include <dlaf/eigensolver/gen_to_std/mc.tpp>

/// ---- ETI
namespace dlaf {

#define DLAF_GENTOSTD_ETI(KWORD, DATATYPE)                                                         \
  KWORD template void Eigensolver<Backend::MC>::genToStd<DATATYPE>(blas::Uplo,                     \
                                                                   Matrix<DATATYPE, Device::CPU>&, \
                                                                   Matrix<DATATYPE, Device::CPU>&);

DLAF_GENTOSTD_ETI(extern, float)
DLAF_GENTOSTD_ETI(extern, double)
DLAF_GENTOSTD_ETI(extern, std::complex<float>)
DLAF_GENTOSTD_ETI(extern, std::complex<double>)

}
