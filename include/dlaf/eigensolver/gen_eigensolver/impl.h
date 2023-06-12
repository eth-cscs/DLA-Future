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

#include <dlaf/eigensolver/eigensolver.h>
#include <dlaf/eigensolver/gen_eigensolver/api.h>
#include <dlaf/eigensolver/gen_to_std.h>
#include <dlaf/factorization/cholesky.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/solver/triangular.h>
#include <dlaf/util_matrix.h>

namespace dlaf::eigensolver::internal {

template <Backend B, Device D, class T>
void GenEigensolver<B, D, T>::call(blas::Uplo uplo, Matrix<T, D>& mat_a, Matrix<T, D>& mat_b,
                                   Matrix<BaseType<T>, D>& eigenvalues, Matrix<T, D>& eigenvectors) {
  factorization::cholesky<B>(uplo, mat_b);
  eigensolver::genToStd<B>(uplo, mat_a, mat_b);

  eigensolver::eigensolver<B>(uplo, mat_a, eigenvalues, eigenvectors);

  solver::triangular<B>(blas::Side::Left, uplo, blas::Op::ConjTrans, blas::Diag::NonUnit, T(1), mat_b,
                        eigenvectors);
}

template <Backend B, Device D, class T>
void GenEigensolver<B, D, T>::call(comm::CommunicatorGrid grid, blas::Uplo uplo, Matrix<T, D>& mat_a,
                                   Matrix<T, D>& mat_b, Matrix<BaseType<T>, D>& eigenvalues,
                                   Matrix<T, D>& eigenvectors) {
  factorization::cholesky<B>(grid, uplo, mat_b);
  eigensolver::genToStd<B>(grid, uplo, mat_a, mat_b);

  eigensolver::eigensolver<B>(grid, uplo, mat_a, eigenvalues, eigenvectors);

  solver::triangular<B>(grid, blas::Side::Left, uplo, blas::Op::ConjTrans, blas::Diag::NonUnit, T(1),
                        mat_b, eigenvectors);
}

}
