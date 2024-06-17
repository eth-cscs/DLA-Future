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

#include <dlaf/eigensolver/eigensolver.h>
#include <dlaf/eigensolver/gen_eigensolver/api.h>
#include <dlaf/eigensolver/gen_to_std.h>
#include <dlaf/factorization/cholesky.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/solver/triangular.h>
#include <dlaf/util_matrix.h>

#include "api.h"

namespace dlaf::eigensolver::internal {

template <Backend B, Device D, class T>
void GenEigensolver<B, D, T>::call(blas::Uplo uplo, Matrix<T, D>& mat_a, Matrix<T, D>& mat_b,
                                   Matrix<BaseType<T>, D>& eigenvalues, Matrix<T, D>& eigenvectors,
                                   const Factorization factorization) {
  if (factorization == Factorization::do_factorization) {
    cholesky_factorization<B>(uplo, mat_b);
  }
  generalized_to_standard<B>(uplo, mat_a, mat_b);

  hermitian_eigensolver<B>(uplo, mat_a, eigenvalues, eigenvectors);

  triangular_solver<B>(blas::Side::Left, uplo, blas::Op::ConjTrans, blas::Diag::NonUnit, T(1), mat_b,
                       eigenvectors);
}

template <Backend B, Device D, class T>
void GenEigensolver<B, D, T>::call(comm::CommunicatorGrid& grid, blas::Uplo uplo, Matrix<T, D>& mat_a,
                                   Matrix<T, D>& mat_b, Matrix<BaseType<T>, D>& eigenvalues,
                                   Matrix<T, D>& eigenvectors, const Factorization factorization) {
  if (factorization == Factorization::do_factorization) {
    cholesky_factorization<B>(grid, uplo, mat_b);
  }
  generalized_to_standard<B>(grid, uplo, mat_a, mat_b);

  hermitian_eigensolver<B>(grid, uplo, mat_a, eigenvalues, eigenvectors);

  triangular_solver<B>(grid, blas::Side::Left, uplo, blas::Op::ConjTrans, blas::Diag::NonUnit, T(1),
                       mat_b, eigenvectors);
}

}
