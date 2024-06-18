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

/// @file

#include <utility>

#include <blas.hh>

#include <dlaf/eigensolver/gen_eigensolver/api.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

#include "gen_eigensolver/api.h"

namespace dlaf {

namespace eigensolver::internal {

template <Backend B, Device D, class T>
void hermitian_generalized_eigensolver_helper(blas::Uplo uplo, Matrix<T, D>& mat_a, Matrix<T, D>& mat_b,
                                              Matrix<BaseType<T>, D>& eigenvalues,
                                              Matrix<T, D>& eigenvectors,
                                              const Factorization factorization) {
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

  eigensolver::internal::GenEigensolver<B, D, T>::call(uplo, mat_a, mat_b, eigenvalues, eigenvectors,
                                                       factorization);
}

template <Backend B, Device D, class T>
EigensolverResult<T, D> hermitian_generalized_eigensolver_helper(blas::Uplo uplo, Matrix<T, D>& mat_a,
                                                                 Matrix<T, D>& mat_b,
                                                                 const Factorization factorization) {
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

  hermitian_generalized_eigensolver_helper<B, D, T>(uplo, mat_a, mat_b, eigenvalues, eigenvectors,
                                                    factorization);

  return {std::move(eigenvalues), std::move(eigenvectors)};
}

template <Backend B, Device D, class T>
void hermitian_generalized_eigensolver_helper(
    comm::CommunicatorGrid& grid, blas::Uplo uplo, Matrix<T, D>& mat_a, Matrix<T, D>& mat_b,
    Matrix<BaseType<T>, D>& eigenvalues, Matrix<T, D>& eigenvectors, const Factorization factorization) {
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

  eigensolver::internal::GenEigensolver<B, D, T>::call(grid, uplo, mat_a, mat_b, eigenvalues,
                                                       eigenvectors, factorization);
}

template <Backend B, Device D, class T>
EigensolverResult<T, D> hermitian_generalized_eigensolver_helper(comm::CommunicatorGrid& grid,
                                                                 blas::Uplo uplo, Matrix<T, D>& mat_a,
                                                                 Matrix<T, D>& mat_b,
                                                                 const Factorization factorization) {
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

  hermitian_generalized_eigensolver_helper<B, D, T>(grid, uplo, mat_a, mat_b, eigenvalues, eigenvectors,
                                                    factorization);

  return {std::move(eigenvalues), std::move(eigenvectors)};
}

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
/// Implementation on local memory.
///
/// @param uplo specifies if upper or lower triangular part of @p mat_a and @p mat_b will be referenced
///
/// @param mat_a contains the Hermitian matrix A
/// @pre @p mat_a is not distributed
/// @pre @p mat_a has size (N x N)
/// @pre @p mat_a has blocksize (NB x NB)
/// @pre @p mat_a has tilesize (NB x NB)
///
/// @param mat_b contains the Hermitian positive definite matrix B
/// @pre @p mat_b is not distributed
/// @pre @p mat_b has size (N x N)
/// @pre @p mat_b has blocksize (NB x NB)
/// @pre @p mat_b has tilesize (NB x NB)
///
/// @param[out] eigenvalues contains the eigenvalues
/// @pre @p eigenvalues is not distributed
/// @pre @p eigenvalues has size (N x 1)
/// @pre @p eigenvalues has blocksize (NB x NB)
/// @pre @p eigenvalues has tilesize (NB x NB)
///
/// @param[out] eigenvectors contains the eigenvectors
/// @pre @p eigenvectors is not distributed
/// @pre @p eigenvectors has size (N x N)
/// @pre @p eigenvectors has blocksize (NB x NB)
/// @pre @p eigenvectors has tilesize (NB x NB)
template <Backend B, Device D, class T>
void hermitian_generalized_eigensolver(blas::Uplo uplo, Matrix<T, D>& mat_a, Matrix<T, D>& mat_b,
                                       Matrix<BaseType<T>, D>& eigenvalues, Matrix<T, D>& eigenvectors) {
  using eigensolver::internal::Factorization;

  eigensolver::internal::hermitian_generalized_eigensolver_helper<B, D, T>(
      uplo, mat_a, mat_b, eigenvalues, eigenvectors, Factorization::do_factorization);
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
///
/// @param mat_a contains the Hermitian matrix A
/// @pre @p mat_a is not distributed
/// @pre @p mat_a has size (N x N)
/// @pre @p mat_a has blocksize (NB x NB)
/// @pre @p mat_a has tilesize (NB x NB)
///
/// @param mat_b contains the Hermitian positive definite matrix B
/// @pre @p mat_b is not distributed
/// @pre @p mat_b has size (N x N)
/// @pre @p mat_b has blocksize (NB x NB)
/// @pre @p mat_b has tilesize (NB x NB)
template <Backend B, Device D, class T>
EigensolverResult<T, D> hermitian_generalized_eigensolver(blas::Uplo uplo, Matrix<T, D>& mat_a,
                                                          Matrix<T, D>& mat_b) {
  using eigensolver::internal::Factorization;

  return eigensolver::internal::hermitian_generalized_eigensolver_helper<B, D, T>(
      uplo, mat_a, mat_b, Factorization::do_factorization);
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
/// Implementation on local memory.
///
/// @param uplo specifies if upper or lower triangular part of @p mat_a and @p mat_b will be referenced
///
/// @param mat_a contains the Hermitian matrix A
/// @pre @p mat_a is not distributed
/// @pre @p mat_a has size (N x N)
/// @pre @p mat_a has blocksize (NB x NB)
/// @pre @p mat_a has tilesize (NB x NB)
///
/// @param mat_b contains the Cholesky factorisation of the Hermitian positive definite matrix B
/// @pre @p mat_b is not distributed
/// @pre @p mat_b has size (N x N)
/// @pre @p mat_b has blocksize (NB x NB)
/// @pre @p mat_b has tilesize (NB x NB)
/// @pre @p mat_b is the result of a Cholesky factorization
///
/// @param[out] eigenvalues contains the eigenvalues
/// @pre @p eigenvalues is not distributed
/// @pre @p eigenvalues has size (N x 1)
/// @pre @p eigenvalues has blocksize (NB x NB)
/// @pre @p eigenvalues has tilesize (NB x NB)
///
/// @param[out] eigenvectors contains the eigenvectors
/// @pre @p eigenvectors is not distributed
/// @pre @p eigenvectors has size (N x N)
/// @pre @p eigenvectors has blocksize (NB x NB)
/// @pre @p eigenvectors has tilesize (NB x NB)
template <Backend B, Device D, class T>
void hermitian_generalized_eigensolver_factorized(blas::Uplo uplo, Matrix<T, D>& mat_a,
                                                  Matrix<T, D>& mat_b,
                                                  Matrix<BaseType<T>, D>& eigenvalues,
                                                  Matrix<T, D>& eigenvectors) {
  using eigensolver::internal::Factorization;

  eigensolver::internal::hermitian_generalized_eigensolver_helper<B, D, T>(
      uplo, mat_a, mat_b, eigenvalues, eigenvectors, Factorization::already_factorized);
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
///
/// @param mat_a contains the Hermitian matrix A
/// @pre @p mat_a is not distributed
/// @pre @p mat_a has size (N x N)
/// @pre @p mat_a has blocksize (NB x NB)
/// @pre @p mat_a has tilesize (NB x NB)
///
/// @param mat_b contains the Cholesky factorisation of the Hermitian positive definite matrix B
/// @pre @p mat_b is not distributed
/// @pre @p mat_b has size (N x N)
/// @pre @p mat_b has blocksize (NB x NB)
/// @pre @p mat_b has tilesize (NB x NB)
/// @pre @p mat_b is the result of a Cholesky factorization
template <Backend B, Device D, class T>
EigensolverResult<T, D> hermitian_generalized_eigensolver_factorized(blas::Uplo uplo,
                                                                     Matrix<T, D>& mat_a,
                                                                     Matrix<T, D>& mat_b) {
  using eigensolver::internal::Factorization;

  return eigensolver::internal::hermitian_generalized_eigensolver_helper<B, D, T>(
      uplo, mat_a, mat_b, Factorization::already_factorized);
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
///
/// @param mat_a contains the Hermitian matrix A
/// @pre @p mat_a is distributed according to @p grid
/// @pre @p mat_a has size (N x N)
/// @pre @p mat_a has blocksize (NB x NB)
/// @pre @p mat_a has tilesize (NB x NB)
///
/// @param mat_b contains the Hermitian positive definite matrix B
/// @pre @p mat_b is distributed according to @p grid
/// @pre @p mat_b has size (N x N)
/// @pre @p mat_b has blocksize (NB x NB)
/// @pre @p mat_b has tilesize (NB x NB)
///
/// @param eigenvalues is a N x 1 matrix which on output contains the eigenvalues
/// @pre @p eigenvalues is not distributed
/// @pre @p eigenvalues has size (N x 1)
/// @pre @p eigenvalues has blocksize (NB x 1)
/// @pre @p eigenvalues has tilesize (NB x 1)
///
/// @param[out] eigenvectors contains the eigenvectors
/// @pre @p eigenvectors is distributed according to @p grid
/// @pre @p eigenvectors has size (N x N)
/// @pre @p eigenvectors has blocksize (NB x NB)
/// @pre @p eigenvectors has tilesize (NB x NB)
template <Backend B, Device D, class T>
void hermitian_generalized_eigensolver(comm::CommunicatorGrid& grid, blas::Uplo uplo,
                                       Matrix<T, D>& mat_a, Matrix<T, D>& mat_b,
                                       Matrix<BaseType<T>, D>& eigenvalues, Matrix<T, D>& eigenvectors) {
  using eigensolver::internal::Factorization;

  eigensolver::internal::hermitian_generalized_eigensolver_helper<B, D, T>(
      grid, uplo, mat_a, mat_b, eigenvalues, eigenvectors, Factorization::do_factorization);
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
///
/// @param mat_a contains the Hermitian matrix A
/// @pre @p mat_a is distributed according to @p grid
/// @pre @p mat_a has size (N x N)
/// @pre @p mat_a has blocksize (NB x NB)
/// @pre @p mat_a has tilesize (NB x NB)
///
/// @param mat_b contains the Hermitian positive definite matrix B
/// @pre @p mat_b is distributed according to @p grid
/// @pre @p mat_b has size (N x N)
/// @pre @p mat_b has blocksize (NB x NB)
/// @pre @p mat_b has tilesize (NB x NB)
template <Backend B, Device D, class T>
EigensolverResult<T, D> hermitian_generalized_eigensolver(comm::CommunicatorGrid& grid, blas::Uplo uplo,
                                                          Matrix<T, D>& mat_a, Matrix<T, D>& mat_b) {
  using eigensolver::internal::Factorization;

  return eigensolver::internal::hermitian_generalized_eigensolver_helper<B, D, T>(
      grid, uplo, mat_a, mat_b, Factorization::do_factorization);
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
///
/// @param mat_a contains the Hermitian matrix A
/// @pre @p mat_a is distributed according to @p grid
/// @pre @p mat_a has size (N x N)
/// @pre @p mat_a has blocksize (NB x NB)
/// @pre @p mat_a has tilesize (NB x NB)
///
/// @param mat_b contains the Cholesky factorisation of the Hermitian positive definite matrix B
/// @pre @p mat_b is distributed according to @p grid
/// @pre @p mat_b has size (N x N)
/// @pre @p mat_b has blocksize (NB x NB)
/// @pre @p mat_b has tilesize (NB x NB)
/// @pre @p mat_b is the result of a Cholesky factorization
///
/// @param eigenvalues is a N x 1 matrix which on output contains the eigenvalues
/// @pre @p eigenvalues is not distributed
/// @pre @p eigenvalues has size (N x 1)
/// @pre @p eigenvalues has blocksize (NB x 1)
/// @pre @p eigenvalues has tilesize (NB x 1)
///
/// @param[out] eigenvectors contains the eigenvectors
/// @pre @p eigenvectors is distributed according to @p grid
/// @pre @p eigenvectors has size (N x N)
/// @pre @p eigenvectors has blocksize (NB x NB)
/// @pre @p eigenvectors has tilesize (NB x NB)
template <Backend B, Device D, class T>
void hermitian_generalized_eigensolver_factorized(comm::CommunicatorGrid& grid, blas::Uplo uplo,
                                                  Matrix<T, D>& mat_a, Matrix<T, D>& mat_b,
                                                  Matrix<BaseType<T>, D>& eigenvalues,
                                                  Matrix<T, D>& eigenvectors) {
  using eigensolver::internal::Factorization;
  eigensolver::internal::hermitian_generalized_eigensolver_helper<B, D, T>(
      grid, uplo, mat_a, mat_b, eigenvalues, eigenvectors, Factorization::already_factorized);
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
///
/// @param mat_a contains the Hermitian matrix A
/// @pre @p mat_a is distributed according to @p grid
/// @pre @p mat_a has size (N x N)
/// @pre @p mat_a has blocksize (NB x NB)
/// @pre @p mat_a has tilesize (NB x NB)
///
/// @param mat_b contains the Cholesky factorisation of the Hermitian positive definite matrix B
/// @pre @p mat_b is distributed according to @p grid
/// @pre @p mat_b has size (N x N)
/// @pre @p mat_b has blocksize (NB x NB)
/// @pre @p mat_b has tilesize (NB x NB)
/// @pre @p mat_b is the result of a Cholesky factorization
template <Backend B, Device D, class T>
EigensolverResult<T, D> hermitian_generalized_eigensolver_factorized(
    comm::CommunicatorGrid& grid, blas::Uplo uplo, Matrix<T, D>& mat_a, Matrix<T, D>& mat_b) {
  using eigensolver::internal::Factorization;

  return eigensolver::internal::hermitian_generalized_eigensolver_helper<B, D, T>(
      grid, uplo, mat_a, mat_b, Factorization::already_factorized);
}
}
