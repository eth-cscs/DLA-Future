//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include "dlaf/eigensolver/eigensolver.h"
#include "dlaf/eigensolver/gen_to_std.h"
#include "dlaf/factorization/cholesky.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/solver/triangular.h"
#include "dlaf/util_matrix.h"

#include "dlaf/eigensolver/gen_eigensolver/api.h"

namespace dlaf::eigensolver::internal {

template <Backend B, Device D, class T>
EigensolverResult<T, D> GenEigensolver<B, D, T>::call(blas::Uplo uplo, Matrix<T, D>& mat_a,
                                                      Matrix<T, D>& mat_b) {
  factorization::cholesky<B>(uplo, mat_b);
  eigensolver::genToStd<B>(uplo, mat_a, mat_b);

  auto ret = eigensolver::eigensolver<B>(uplo, mat_a);

  solver::triangular<B>(blas::Side::Left, uplo, blas::Op::ConjTrans, blas::Diag::NonUnit, T(1), mat_b,
                        ret.eigenvectors);

  return ret;
}

}
