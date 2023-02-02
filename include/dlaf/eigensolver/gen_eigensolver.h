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

#include <blas.hh>
#include "dlaf/eigensolver/gen_eigensolver/api.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf::eigensolver {

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
template <Backend B, Device D, class T>
EigensolverResult<T, D> genEigensolver(blas::Uplo uplo, Matrix<T, D>& mat_a, Matrix<T, D>& mat_b) {
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);
  DLAF_ASSERT(matrix::local_matrix(mat_b), mat_b);
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_size(mat_b), mat_b);
  DLAF_ASSERT(matrix::square_blocksize(mat_b), mat_b);
  DLAF_ASSERT(mat_a.size() == mat_b.size(), mat_a, mat_b);
  DLAF_ASSERT(mat_a.blockSize() == mat_b.blockSize(), mat_a, mat_b);

  return internal::GenEigensolver<B, D, T>::call(uplo, mat_a, mat_b);
}

/// Generalized Eigensolver.
///
/// It solves the generalized eigenvalue problem A * x = lambda * B * x.
///
/// On exit:
/// - the lower triangle or the upper triangle (depending on @p uplo) of @p mat_a,
/// including the diagonal, is destroyed.
/// - @p mat_b contains the Cholesky decomposition of B
///
/// Implementation on distributed memory.
///
/// @return struct ReturnEigensolverType with eigenvalues, as a vector<T>, and eigenvectors as a Matrix
/// @param grid is the communicator grid on which the matrices @p mat_a and @p mat_b have been distributed,
/// @param uplo specifies if upper or lower triangular part of @p mat_a and @p mat_b will be referenced
/// @param mat_a contains the Hermitian matrix A
/// @param mat_b contains the Hermitian positive definite matrix B
template <Backend B, Device D, class T>
EigensolverResult<T, D> genEigensolver(comm::CommunicatorGrid grid, blas::Uplo uplo, Matrix<T, D>& mat_a,
                                       Matrix<T, D>& mat_b) {
  DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);
  DLAF_ASSERT(matrix::equal_process_grid(mat_b, grid), mat_b, grid);
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_size(mat_b), mat_b);
  DLAF_ASSERT(matrix::square_blocksize(mat_b), mat_b);
  DLAF_ASSERT(mat_a.size() == mat_b.size(), mat_a, mat_b);
  DLAF_ASSERT(mat_a.blockSize() == mat_b.blockSize(), mat_a, mat_b);

  return internal::GenEigensolver<B, D, T>::call(grid, uplo, mat_a, mat_b);
}
}
