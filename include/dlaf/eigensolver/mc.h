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
  /// obtained from potrf (lower matrix from Cholesky decomposition), solving B=inv(L)*A*inv(L**H),
  /// implementation on local memory.
  ///
  /// @param mat_a on entry it contains the square and Hermitian matrix A, on exit the matrix elements
  /// are overwritten with the elements of the matrix B.
  /// @param mat_l contains the lower triangular matrix L. Only the tiles of the matrix which contain the
  /// lower triangular part are accessed.
  template <class T>
  static void genToStd(Matrix<T, Device::CPU>& mat_a, Matrix<T, Device::CPU>& mat_l);
};

}

#include <dlaf/eigensolver/gen_to_std/mc.tpp>

/// ---- ETI
namespace dlaf {

#define DLAF_GENTOSTD_ETI(KWORD, DATATYPE)                                                         \
  KWORD template void Eigensolver<Backend::MC>::genToStd<DATATYPE>(Matrix<DATATYPE, Device::CPU>&, \
                                                                   Matrix<DATATYPE, Device::CPU>&);

DLAF_GENTOSTD_ETI(extern, float)
DLAF_GENTOSTD_ETI(extern, double)
DLAF_GENTOSTD_ETI(extern, std::complex<float>)
DLAF_GENTOSTD_ETI(extern, std::complex<double>)

}
