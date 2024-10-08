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

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/eigensolver/eigensolver/api.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace dlaf {

/// Standard Eigensolver.
///
/// It solves the standard eigenvalue problem A * x = lambda * x.
///
/// On exit, the lower triangle or the upper triangle (depending on @p uplo) of @p mat_a,
/// including the diagonal, is destroyed. @p eigenvalues will contain all the eigenvalues
/// lambda, while @p eigenvectors will contain all the corresponding eigenvectors x.
///
/// Implementation on local memory.
///
/// @param uplo specifies if upper or lower triangular part of @p mat will be referenced
///
/// @param[in,out] mat contains the Hermitian matrix A
/// @pre @p mat is not distributed
/// @pre @p mat has size (N x N)
/// @pre @p mat has blocksize (NB x NB)
/// @pre @p mat has tilesize (NB x NB)
///
/// @param[out] eigenvalues contains the eigenvalues
/// @pre @p eigenvalues is not distributed
/// @pre @p eigenvalues has size (N x 1)
/// @pre @p eigenvalues has blocksize (NB x 1)
/// @pre @p eigenvalues has tilesize (NB x 1)
///
/// @param[out] eigenvectors contains the eigenvectors
/// @pre @p eigenvectors is not distributed
/// @pre @p eigenvectors has size (N x N)
/// @pre @p eigenvectors has blocksize (NB x NB)
/// @pre @p eigenvectors has tilesize (NB x NB)
///
/// @param[in] eigenvalues_index_begin is the index of the first eigenvalue to compute
/// @pre @p eigenvalues_index_begin == 0
/// @param[in] eigenvalues_index_end is the index of the last eigenvalue to compute (exclusive)
/// @pre @p eigenvalues_index_begin <= eigenvalues_index_end < N
template <Backend B, Device D, class T>
void hermitian_eigensolver(blas::Uplo uplo, Matrix<T, D>& mat, Matrix<BaseType<T>, D>& eigenvalues,
                           Matrix<T, D>& eigenvectors, const SizeType eigenvalues_index_begin,
                           const SizeType eigenvalues_index_end) {
  DLAF_ASSERT(matrix::local_matrix(mat), mat);
  DLAF_ASSERT(square_size(mat), mat);
  DLAF_ASSERT(square_blocksize(mat), mat);
  DLAF_ASSERT(matrix::local_matrix(eigenvalues), eigenvalues);
  DLAF_ASSERT(eigenvalues.size().rows() == eigenvectors.size().rows(), eigenvalues, eigenvectors);
  DLAF_ASSERT(eigenvalues.blockSize().rows() == eigenvectors.blockSize().rows(), eigenvalues,
              eigenvectors);
  DLAF_ASSERT(matrix::local_matrix(eigenvectors), eigenvectors);
  DLAF_ASSERT(square_size(eigenvectors), eigenvectors);
  DLAF_ASSERT(square_blocksize(eigenvectors), eigenvectors);
  DLAF_ASSERT(eigenvectors.size() == mat.size(), eigenvectors, mat);
  DLAF_ASSERT(eigenvectors.blockSize() == mat.blockSize(), eigenvectors, mat);
  DLAF_ASSERT(single_tile_per_block(mat), mat);
  DLAF_ASSERT(single_tile_per_block(eigenvalues), eigenvalues);
  DLAF_ASSERT(single_tile_per_block(eigenvectors), eigenvectors);
  DLAF_ASSERT(eigenvalues_index_begin == 0, eigenvalues_index_begin);
  DLAF_ASSERT(eigenvalues_index_end >= eigenvalues_index_begin, eigenvalues_index_end,
              eigenvalues_index_begin);
  DLAF_ASSERT(eigenvalues_index_end <= mat.size().rows(), eigenvalues_index_end, mat.size().rows());

  eigensolver::internal::Eigensolver<B, D, T>::call(uplo, mat, eigenvalues, eigenvectors,
                                                    eigenvalues_index_begin, eigenvalues_index_end);
}

/// Standard Eigensolver.
///
/// It solves the standard eigenvalue problem A * x = lambda * x.
///
/// On exit, the lower triangle or the upper triangle (depending on @p uplo) of @p mat_a,
/// including the diagonal, is destroyed. @p eigenvalues will contain all the eigenvalues
/// lambda, while @p eigenvectors will contain all the corresponding eigenvectors x.
///
/// Implementation on local memory.
///
/// @param uplo specifies if upper or lower triangular part of @p mat will be referenced
///
/// @param[in,out] mat contains the Hermitian matrix A
/// @pre @p mat is not distributed
/// @pre @p mat has size (N x N)
/// @pre @p mat has blocksize (NB x NB)
/// @pre @p mat has tilesize (NB x NB)
///
/// @param[out] eigenvalues contains the eigenvalues
/// @pre @p eigenvalues is not distributed
/// @pre @p eigenvalues has size (N x 1)
/// @pre @p eigenvalues has blocksize (NB x 1)
/// @pre @p eigenvalues has tilesize (NB x 1)
///
/// @param[out] eigenvectors contains the eigenvectors
/// @pre @p eigenvectors is not distributed
/// @pre @p eigenvectors has size (N x N)
/// @pre @p eigenvectors has blocksize (NB x NB)
/// @pre @p eigenvectors has tilesize (NB x NB)
template <Backend B, Device D, class T>
void hermitian_eigensolver(blas::Uplo uplo, Matrix<T, D>& mat, Matrix<BaseType<T>, D>& eigenvalues,
                           Matrix<T, D>& eigenvectors) {
  hermitian_eigensolver<B, D, T>(uplo, mat, eigenvalues, eigenvectors, 0l, mat.size().rows());
}

/// Standard Eigensolver.
///
/// It solves the standard eigenvalue problem A * x = lambda * x.
///
/// On exit, the lower triangle or the upper triangle (depending on @p uplo) of @p mat_a,
/// including the diagonal, is destroyed.
///
/// Implementation on local memory.
///
/// @return ReturnEigensolverType with eigenvalues and eigenvectors as a Matrix
///
/// @param uplo specifies if upper or lower triangular part of @p mat will be referenced
///
/// @param[in,out] mat contains the Hermitian matrix A
/// @pre @p mat is not distributed
/// @pre @p mat has size (N x N)
/// @pre @p mat has blocksize (NB x NB)
/// @pre @p mat has tilesize (NB x NB)
///
/// @param[in] eigenvalues_index_begin is the index of the first eigenvalue to compute
/// @pre @p eigenvalues_index_begin == 0
/// @param[in] eigenvalues_index_end is the index of the last eigenvalue to compute (exclusive)
/// @pre @p eigenvalues_index_end < N
template <Backend B, Device D, class T>
EigensolverResult<T, D> hermitian_eigensolver(blas::Uplo uplo, Matrix<T, D>& mat,
                                              const SizeType eigenvalues_index_begin,
                                              const SizeType eigenvalues_index_end) {
  const SizeType size = mat.size().rows();
  matrix::Matrix<BaseType<T>, D> eigenvalues(LocalElementSize(size, 1),
                                             TileElementSize(mat.blockSize().rows(), 1));
  matrix::Matrix<T, D> eigenvectors(LocalElementSize(size, size), mat.blockSize());

  hermitian_eigensolver<B, D, T>(uplo, mat, eigenvalues, eigenvectors, eigenvalues_index_begin,
                                 eigenvalues_index_end);
  return {std::move(eigenvalues), std::move(eigenvectors)};
}

/// Standard Eigensolver.
///
/// It solves the standard eigenvalue problem A * x = lambda * x.
///
/// On exit, the lower triangle or the upper triangle (depending on @p uplo) of @p mat_a,
/// including the diagonal, is destroyed.
///
/// Implementation on local memory.
///
/// @return ReturnEigensolverType with eigenvalues and eigenvectors as a Matrix
///
/// @param uplo specifies if upper or lower triangular part of @p mat will be referenced
///
/// @param[in,out] mat contains the Hermitian matrix A
/// @pre @p mat is not distributed
/// @pre @p mat has size (N x N)
/// @pre @p mat has blocksize (NB x NB)
/// @pre @p mat has tilesize (NB x NB)
template <Backend B, Device D, class T>
EigensolverResult<T, D> hermitian_eigensolver(blas::Uplo uplo, Matrix<T, D>& mat) {
  return hermitian_eigensolver<B, D, T>(uplo, mat, 0l, mat.size().rows());
}

/// Standard Eigensolver.
///
/// It solves the standard eigenvalue problem A * x = lambda * x.
///
/// On exit, the lower triangle or the upper triangle (depending on @p uplo) of @p mat_a,
/// including the diagonal, is destroyed. @p eigenvalues will contain all the eigenvalues
/// lambda, while @p eigenvectors will contain all the corresponding eigenvectors x.
///
/// Implementation on distributed memory.
///
/// @param grid is the communicator grid on which the matrix @p mat has been distributed
///
/// @param uplo specifies if upper or lower triangular part of @p mat will be referenced
///
/// @param[in,out] mat contains the Hermitian matrix A
/// @pre @p mat is distributed according to @p grid
/// @pre @p mat has size (N x N)
/// @pre @p mat has blocksize (NB x NB)
/// @pre @p mat has tilesize (NB x NB)
///
/// @param[out] eigenvalues contains the eigenvalues
/// @pre @p eigenvalues is stored on all ranks
/// @pre @p eigenvalues has size (N x 1)
/// @pre @p eigenvalues has blocksize (NB x 1)
/// @pre @p eigenvalues has tilesize (NB x 1)
///
/// @param[out] eigenvectors contains the eigenvectors
/// @pre @p eigenvectors is distributed according to @p grid
/// @pre @p eigenvectors has size (N x N)
/// @pre @p eigenvectors has blocksize (NB x NB)
/// @pre @p eigenvectors has tilesize (NB x NB)
///
/// @param[in] eigenvalues_index_begin is the index of the first eigenvalue to compute
/// @pre @p eigenvalues_index_begin == 0
/// @param[in] eigenvalues_index_end is the index of the last eigenvalue to compute (exclusive)
/// @pre @p eigenvalues_index_end < N
template <Backend B, Device D, class T>
void hermitian_eigensolver(comm::CommunicatorGrid& grid, blas::Uplo uplo, Matrix<T, D>& mat,
                           Matrix<BaseType<T>, D>& eigenvalues, Matrix<T, D>& eigenvectors,
                           const SizeType eigenvalues_index_begin,
                           const SizeType eigenvalues_index_end) {
  DLAF_ASSERT(matrix::equal_process_grid(mat, grid), mat);
  DLAF_ASSERT(square_size(mat), mat);
  DLAF_ASSERT(square_blocksize(mat), mat);
  DLAF_ASSERT(matrix::local_matrix(eigenvalues), eigenvalues);
  DLAF_ASSERT(eigenvalues.size().rows() == eigenvectors.size().rows(), eigenvalues, eigenvectors);
  DLAF_ASSERT(eigenvalues.blockSize().rows() == eigenvectors.blockSize().rows(), eigenvalues,
              eigenvectors);
  DLAF_ASSERT(matrix::equal_process_grid(eigenvectors, grid), eigenvectors);
  DLAF_ASSERT(square_size(eigenvectors), eigenvectors);
  DLAF_ASSERT(square_blocksize(eigenvectors), eigenvectors);
  DLAF_ASSERT(eigenvectors.size() == mat.size(), eigenvectors, mat);
  DLAF_ASSERT(eigenvectors.blockSize() == mat.blockSize(), eigenvectors, mat);
  DLAF_ASSERT(single_tile_per_block(mat), mat);
  DLAF_ASSERT(single_tile_per_block(eigenvalues), eigenvalues);
  DLAF_ASSERT(single_tile_per_block(eigenvectors), eigenvectors);
  DLAF_ASSERT(eigenvalues_index_begin == 0, eigenvalues_index_begin);
  DLAF_ASSERT(eigenvalues_index_end >= eigenvalues_index_begin, eigenvalues_index_end,
              eigenvalues_index_begin);
  DLAF_ASSERT(eigenvalues_index_end <= mat.size().rows(), eigenvalues_index_end, mat.size().rows());

  eigensolver::internal::Eigensolver<B, D, T>::call(grid, uplo, mat, eigenvalues, eigenvectors,
                                                    eigenvalues_index_begin, eigenvalues_index_end);
}

/// Standard Eigensolver.
///
/// It solves the standard eigenvalue problem A * x = lambda * x.
///
/// On exit, the lower triangle or the upper triangle (depending on @p uplo) of @p mat_a,
/// including the diagonal, is destroyed. @p eigenvalues will contain all the eigenvalues
/// lambda, while @p eigenvectors will contain all the corresponding eigenvectors x.
///
/// Implementation on distributed memory.
///
/// @param grid is the communicator grid on which the matrix @p mat has been distributed
///
/// @param uplo specifies if upper or lower triangular part of @p mat will be referenced
///
/// @param[in,out] mat contains the Hermitian matrix A
/// @pre @p mat is distributed according to @p grid
/// @pre @p mat has size (N x N)
/// @pre @p mat has blocksize (NB x NB)
/// @pre @p mat has tilesize (NB x NB)
///
/// @param[out] eigenvalues contains the eigenvalues
/// @pre @p eigenvalues is stored on all ranks
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
void hermitian_eigensolver(comm::CommunicatorGrid& grid, blas::Uplo uplo, Matrix<T, D>& mat,
                           Matrix<BaseType<T>, D>& eigenvalues, Matrix<T, D>& eigenvectors) {
  hermitian_eigensolver<B, D, T>(grid, uplo, mat, eigenvalues, eigenvectors, 0l, mat.size().rows());
}

/// Standard Eigensolver.
///
/// It solves the standard eigenvalue problem A * x = lambda * x.
///
/// On exit, the lower triangle or the upper triangle (depending on @p uplo) of @p mat_a,
/// including the diagonal, is destroyed.
///
/// Implementation on distributed memory.
///
/// @return struct ReturnEigensolverType with eigenvalues and eigenvectors as a Matrix
///
/// @param grid is the communicator grid on which the matrix @p mat has been distributed
///
/// @param uplo specifies if upper or lower triangular part of @p mat will be referenced
///
/// @param[in,out] mat contains the Hermitian matrix A
/// @pre @p mat is distributed according to @p grid
/// @pre @p mat has size (N x N)
/// @pre @p mat has blocksize (NB x NB)
/// @pre @p mat has tilesize (NB x NB)
///
/// @param[in] eigenvalues_index_begin is the index of the first eigenvalue to compute
/// @pre @p eigenvalues_index_begin == 0
/// @param[in] eigenvalues_index_end is the index of the last eigenvalue to compute (exclusive)
/// @pre @p eigenvalues_index_end < N
template <Backend B, Device D, class T>
EigensolverResult<T, D> hermitian_eigensolver(comm::CommunicatorGrid& grid, blas::Uplo uplo,
                                              Matrix<T, D>& mat, const SizeType eigenvalues_index_begin,
                                              const SizeType eigenvalues_index_end) {
  const SizeType size = mat.size().rows();
  matrix::Matrix<BaseType<T>, D> eigenvalues(LocalElementSize(size, 1),
                                             TileElementSize(mat.blockSize().rows(), 1));
  matrix::Matrix<T, D> eigenvectors(GlobalElementSize(size, size), mat.blockSize(), grid);

  hermitian_eigensolver<B, D, T>(grid, uplo, mat, eigenvalues, eigenvectors, eigenvalues_index_begin,
                                 eigenvalues_index_end);
  return {std::move(eigenvalues), std::move(eigenvectors)};
}

/// Standard Eigensolver.
///
/// It solves the standard eigenvalue problem A * x = lambda * x.
///
/// On exit, the lower triangle or the upper triangle (depending on @p uplo) of @p mat_a,
/// including the diagonal, is destroyed.
///
/// Implementation on distributed memory.
///
/// @return struct ReturnEigensolverType with eigenvalues and eigenvectors as a Matrix
///
/// @param grid is the communicator grid on which the matrix @p mat has been distributed
///
/// @param uplo specifies if upper or lower triangular part of @p mat will be referenced
///
/// @param[in,out] mat contains the Hermitian matrix A
/// @pre @p mat is distributed according to @p grid
/// @pre @p mat has size (N x N)
/// @pre @p mat has blocksize (NB x NB)
/// @pre @p mat has tilesize (NB x NB)
template <Backend B, Device D, class T>
EigensolverResult<T, D> hermitian_eigensolver(comm::CommunicatorGrid& grid, blas::Uplo uplo,
                                              Matrix<T, D>& mat) {
  return hermitian_eigensolver<B, D, T>(grid, uplo, mat, 0l, mat.size().rows());
}
}
