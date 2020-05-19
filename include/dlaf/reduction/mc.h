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
#include "dlaf/reduction/internal.h"
#include "dlaf/types.h"

namespace dlaf {

template <>
struct Reduction<Backend::MC> {
  /// Reduce a Hermitian definite generalized eigenproblem to standard form, using the factorization
  /// obtained from potrf (lower matrix from Cholesky decomposition), solving B=inv(L)*A*inv(L**H),
  /// implementation on local memory.
  ///
  /// @param mat_a on entry it contains the square and Hermitian matrix A, on exit the matrix elements
  /// are overwritten with the elements of the matrix B.
  /// @param mat_l contains the lower triangular matrix L. Only the tiles of the matrix which contain the
  /// lower triangular part are accessed.
  template <class T>
  static void stdeigen(Matrix<T, Device::CPU>& mat_a, Matrix<const T, Device::CPU>& mat_l);

  /// Reduce a Hermitian definite generalized eigenproblem to standard form, using the factorization
  /// obtained from potrf (lower matrix from Cholesky decomposition), solving B=inv(L)*A*inv(L**H),
  /// implementation on distributed memory.
  ///
  /// @param mat_a on entry it contains the square and Hermitian matrix A, on exit the matrix elements
  /// are overwritten with the elements of the matrix B.
  /// @param mat_l contains the lower triangular matrix L. Only the tiles of the matrix which contain the
  /// lower triangular part are accessed.
  template <class T>
  static void stdeigen(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_a,
                       Matrix<const T, Device::CPU>& mat_l);
};

}

#include <dlaf/reduction/stdeigen/mc.tpp>
