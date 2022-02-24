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

#include <blas.hh>
#include "dlaf/eigensolver/gen_eigensolver/impl.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {

/// Generalized Eigensolver.
///
/// It solves the generalized eigenvalue problem A * x = lambda * B * x.
///
/// On exit:
/// - the lower triangle or the upper triangle (depending on @p uplo) of @p mat_a,
/// including the diagonal, is destroyed.
/// - @p mat_b contains the Cholesky decomposition of B
///
/// Implementation on local memory.
///
/// @return struct ReturnEigensolverType with eigenvalues, as a vector<T>, and eigenvectors as a Matrix
/// @param uplo specifies if upper or lower triangular part of @p mat_a and @p mat_b will be referenced
/// @param mat_a contains the Hermitian matrix A
/// @param mat_b contains the Hermitian positive definite matrix B
template <Backend backend, Device device, class T>
EigensolverResult<T, device> genEigensolver(blas::Uplo uplo, Matrix<T, device>& mat_a,
                                            Matrix<T, device>& mat_b) {
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);
  DLAF_ASSERT(matrix::local_matrix(mat_b), mat_b);
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_size(mat_b), mat_b);
  DLAF_ASSERT(matrix::square_blocksize(mat_b), mat_b);
  DLAF_ASSERT(mat_a.size() == mat_b.size(), mat_a, mat_b);
  DLAF_ASSERT(mat_a.blockSize() == mat_b.blockSize(), mat_a, mat_b);

  return internal::GenEigensolver<backend, device, T>::call(uplo, mat_a, mat_b);
}
}
}
