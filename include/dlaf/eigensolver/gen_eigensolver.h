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

/// @file

#include <blas.hh>

#include <dlaf/eigensolver/gen_eigensolver/api.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace dlaf::eigensolver {

/// Generalized Eigensolver.
///
/// It solves the generalized eigenvalue problem A * x = lambda * B * x.
///
/// On exit:
/// - the lower triangle or the upper triangle (depending on @p uplo) of @p mat_a,
/// including the diagonal, is destroyed.
/// - @p mat_b contains the Cholesky decomposition of B
/// - @p eigenvalues contains all the eigenvalues lambda
/// - @p eigenvectors contains all the eigenvectors x
///
/// Implementation on local memory.
///
/// @param uplo specifies if upper or lower triangular part of @p mat_a and @p mat_b will be referenced
/// @param mat_a contains the Hermitian matrix A
/// @param mat_b contains the Hermitian positive definite matrix B
/// @param eigenvalues is a N x 1 matrix which on output contains the eigenvalues
/// @param eigenvectors is a N x N matrix which on output contains the eigenvectors
/// @pre mat_a is not distributed
/// @pre mat_a has a square size
/// @pre mat_a has a square blocksize
/// @pre mat_a has equal tile and block sizes
/// @pre mat_b is not distributed
/// @pre mat_b has a square size
/// @pre mat_b has a square blocksize
/// @pre mat_b has equal tile and block sizes
/// @pre eigenvalues is not distributed
/// @pre eigenvalues has equal tile and block sizes
/// @pre eigenvectors is not distributed
/// @pre eigenvectors has a square blocksize
/// @pre eigenvectors has equal tile and block sizes
template <Backend B, Device D, class T>
void genEigensolver(blas::Uplo uplo, Matrix<T, D>& mat_a, Matrix<T, D>& mat_b,
                    Matrix<BaseType<T>, D>& eigenvalues, Matrix<T, D>& eigenvectors) {
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);
  DLAF_ASSERT(matrix::local_matrix(mat_b), mat_b);
  DLAF_ASSERT(matrix::local_matrix(eigenvalues), eigenvalues);
  DLAF_ASSERT(matrix::local_matrix(eigenvectors), eigenvectors);
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_size(mat_b), mat_b);
  DLAF_ASSERT(matrix::square_blocksize(mat_b), mat_b);
  DLAF_ASSERT(matrix::square_size(eigenvectors), eigenvectors);
  DLAF_ASSERT(matrix::square_blocksize(eigenvectors), eigenvectors);
  DLAF_ASSERT(mat_a.size() == mat_b.size(), mat_a, mat_b);
  DLAF_ASSERT(mat_a.blockSize() == mat_b.blockSize(), mat_a, mat_b);
  DLAF_ASSERT(eigenvalues.size().rows() == eigenvectors.size().rows(), eigenvalues, eigenvectors);
  DLAF_ASSERT(eigenvalues.blockSize().rows() == eigenvectors.blockSize().rows(), eigenvalues,
              eigenvectors);
  DLAF_ASSERT(eigenvectors.size() == mat_a.size(), eigenvectors, mat_a);
  DLAF_ASSERT(eigenvectors.blockSize() == mat_a.blockSize(), eigenvectors, mat_a);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_a), mat_a);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_b), mat_b);
  DLAF_ASSERT(matrix::single_tile_per_block(eigenvalues), eigenvalues);
  DLAF_ASSERT(matrix::single_tile_per_block(eigenvectors), eigenvectors);

  internal::GenEigensolver<B, D, T>::call(uplo, mat_a, mat_b, eigenvalues, eigenvectors);
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
/// Implementation on local memory.
///
/// @return struct ReturnEigensolverType with eigenvalues, as a vector<T>, and eigenvectors as a Matrix
/// @param uplo specifies if upper or lower triangular part of @p mat_a and @p mat_b will be referenced
/// @param mat_a contains the Hermitian matrix A
/// @param mat_b contains the Hermitian positive definite matrix B
/// @pre mat_a is not distributed
/// @pre mat_a has a square size
/// @pre mat_a has a square blocksize
/// @pre mat_a has equal tile and block sizes
/// @pre mat_b is not distributed
/// @pre mat_b has a square size
/// @pre mat_b has a square blocksize
/// @pre mat_b has equal tile and block sizes
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

  const SizeType size = mat_a.size().rows();

  matrix::Matrix<BaseType<T>, D> eigenvalues(LocalElementSize(size, 1),
                                             TileElementSize(mat_a.blockSize().rows(), 1));
  matrix::Matrix<T, D> eigenvectors(LocalElementSize(size, size), mat_a.blockSize());

  genEigensolver<B, D, T>(uplo, mat_a, mat_b, eigenvalues, eigenvectors);

  return {std::move(eigenvalues), std::move(eigenvectors)};
}

/// Generalized Eigensolver.
///
/// It solves the generalized eigenvalue problem A * x = lambda * B * x.
///
/// On exit:
/// - the lower triangle or the upper triangle (depending on @p uplo) of @p mat_a,
/// including the diagonal, is destroyed.
/// - @p mat_b contains the Cholesky decomposition of B
/// - @p eigenvalues contains all the eigenvalues lambda
/// - @p eigenvectors contains all the eigenvectors x
///
/// Implementation on distributed memory.
///
/// @param grid is the communicator grid on which the matrices @p mat_a and @p mat_b have been distributed,
/// @param uplo specifies if upper or lower triangular part of @p mat_a and @p mat_b will be referenced
/// @param mat_a contains the Hermitian matrix A
/// @param mat_b contains the Hermitian positive definite matrix B
/// @param eigenvalues is a N x 1 matrix which on output contains the eigenvalues
/// @param eigenvectors is a N x N matrix which on output contains the eigenvectors
/// @pre mat_a is distributed according to grid
/// @pre mat_a has a square size
/// @pre mat_a has a square blocksize
/// @pre mat_a has equal tile and block sizes
/// @pre mat_b is distributed according to grid
/// @pre mat_b has a square size
/// @pre mat_b has a square blocksize
/// @pre mat_b has equal tile and block sizes
/// @pre eigenvalues is distributed according to grid ?? TODO
/// @pre eigenvalues has equal tile and block sizes
/// @pre eigenvectors is distributed according to grid
/// @pre eigenvectors has a square blocksize
/// @pre eigenvectors has equal tile and block sizes
template <Backend B, Device D, class T>
void genEigensolver(comm::CommunicatorGrid grid, blas::Uplo uplo, Matrix<T, D>& mat_a,
                    Matrix<T, D>& mat_b, Matrix<BaseType<T>, D>& eigenvalues,
                    Matrix<T, D>& eigenvectors) {
  DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);
  DLAF_ASSERT(matrix::equal_process_grid(mat_b, grid), mat_b, grid);
  DLAF_ASSERT(matrix::local_matrix(eigenvalues), eigenvalues);
  DLAF_ASSERT(matrix::equal_process_grid(eigenvectors, grid), eigenvectors, grid);
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_size(mat_b), mat_b);
  DLAF_ASSERT(matrix::square_blocksize(mat_b), mat_b);
  DLAF_ASSERT(matrix::square_size(eigenvectors), eigenvectors);
  DLAF_ASSERT(matrix::square_blocksize(eigenvectors), eigenvectors);
  DLAF_ASSERT(mat_a.size() == mat_b.size(), mat_a, mat_b);
  DLAF_ASSERT(mat_a.blockSize() == mat_b.blockSize(), mat_a, mat_b);
  DLAF_ASSERT(eigenvalues.size().rows() == eigenvectors.size().rows(), eigenvalues, eigenvectors);
  DLAF_ASSERT(eigenvalues.blockSize().rows() == eigenvectors.blockSize().rows(), eigenvalues,
              eigenvectors);
  DLAF_ASSERT(eigenvectors.size() == mat_a.size(), eigenvectors, mat_a);
  DLAF_ASSERT(eigenvectors.blockSize() == mat_a.blockSize(), eigenvectors, mat_a);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_a), mat_a);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_b), mat_b);
  DLAF_ASSERT(matrix::single_tile_per_block(eigenvalues), eigenvalues);
  DLAF_ASSERT(matrix::single_tile_per_block(eigenvectors), eigenvectors);

  internal::GenEigensolver<B, D, T>::call(grid, uplo, mat_a, mat_b, eigenvalues, eigenvectors);
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
/// @pre mat_a is distributed according to grid
/// @pre mat_a has a square size
/// @pre mat_a has a square blocksize
/// @pre mat_a has equal tile and block sizes
/// @pre mat_b is distributed according to grid
/// @pre mat_b has a square size
/// @pre mat_b has a square blocksize
/// @pre mat_b has equal tile and block sizes
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

  const SizeType size = mat_a.size().rows();

  matrix::Matrix<BaseType<T>, D> eigenvalues(LocalElementSize(size, 1),
                                             TileElementSize(mat_a.blockSize().rows(), 1));
  matrix::Matrix<T, D> eigenvectors(GlobalElementSize(size, size), mat_a.blockSize(), grid);

  genEigensolver<B, D, T>(grid, uplo, mat_a, mat_b, eigenvalues, eigenvectors);

  return {std::move(eigenvalues), std::move(eigenvectors)};
}
}
