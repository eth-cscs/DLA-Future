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

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/eigensolver/eigensolver/api.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace dlaf::eigensolver {

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
/// @pre mat is not distributed
/// @pre mat has size (N x N)
/// @pre mat has blocksize (NB x NB)
/// @pre mat has tilesize (NB x NB)
///
/// @param[out] eigenvalues contains the eigenvalues
/// @pre eigenvalues is not distributed
/// @pre eigenvalues has size (N x 1)
/// @pre eigenvalues has blocksize (NB x 1)
/// @pre eigenvalues has tilesize (NB x 1)
///
/// @param[out] eigenvectors contains the eigenvectors
/// @pre eigenvectors is not distributed
/// @pre eigenvectors has size (N x N)
/// @pre eigenvectors has blocksize (NB x NB)
/// @pre eigenvectors has tilesize (NB x NB)
template <Backend B, Device D, class T>
void eigensolver(blas::Uplo uplo, Matrix<T, D>& mat, Matrix<BaseType<T>, D>& eigenvalues,
                 Matrix<T, D>& eigenvectors) {
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

  internal::Eigensolver<B, D, T>::call(uplo, mat, eigenvalues, eigenvectors);
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
/// @pre mat is not distributed
/// @pre mat has size (N x N)
/// @pre mat has blocksize (NB x NB)
/// @pre mat has tilesize (NB x NB)
template <Backend B, Device D, class T>
EigensolverResult<T, D> eigensolver(blas::Uplo uplo, Matrix<T, D>& mat) {
  const SizeType size = mat.size().rows();
  matrix::Matrix<BaseType<T>, D> eigenvalues(LocalElementSize(size, 1),
                                             TileElementSize(mat.blockSize().rows(), 1));
  matrix::Matrix<T, D> eigenvectors(LocalElementSize(size, size), mat.blockSize());

  eigensolver<B, D, T>(uplo, mat, eigenvalues, eigenvectors);
  return {std::move(eigenvalues), std::move(eigenvectors)};
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
/// @pre grid is an (NG x MG) grid
///
/// @param uplo specifies if upper or lower triangular part of @p mat will be referenced
///
/// @param[in,out] mat contains the Hermitian matrix A
/// @pre mat is distributed according to @p grid
/// @pre mat has size (N x N)
/// @pre mat has blocksize (NB x NB)
/// @pre mat has tilesize (NB x NB)
///
/// @param[out] eigenvalues contains the eigenvalues
/// @pre eigenvalues is stored on all ranks
/// @pre eigenvalues has size (N x 1)
/// @pre eigenvalues has blocksize (NB x 1)
/// @pre eigenvalues has tilesize (NB x 1)
///
/// @param[out] eigenvectors contains the eigenvectors
/// @pre eigenvectors is distributed according to @p grid
/// @pre eigenvectors has size (N x N)
/// @pre eigenvectors has blocksize (NB x NB)
/// @pre eigenvectors has tilesize (NB x NB)
template <Backend B, Device D, class T>
void eigensolver(comm::CommunicatorGrid grid, blas::Uplo uplo, Matrix<T, D>& mat,
                 Matrix<BaseType<T>, D>& eigenvalues, Matrix<T, D>& eigenvectors) {
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

  internal::Eigensolver<B, D, T>::call(grid, uplo, mat, eigenvalues, eigenvectors);
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
/// @pre mat is distributed according to @p grid
/// @pre mat has size (N x N)
/// @pre mat has blocksize (NB x NB)
/// @pre mat has tilesize (NB x NB)
template <Backend B, Device D, class T>
EigensolverResult<T, D> eigensolver(comm::CommunicatorGrid grid, blas::Uplo uplo, Matrix<T, D>& mat) {
  const SizeType size = mat.size().rows();
  matrix::Matrix<BaseType<T>, D> eigenvalues(LocalElementSize(size, 1),
                                             TileElementSize(mat.blockSize().rows(), 1));
  matrix::Matrix<T, D> eigenvectors(GlobalElementSize(size, size), mat.blockSize(), grid);

  eigensolver<B, D, T>(grid, uplo, mat, eigenvalues, eigenvectors);
  return {std::move(eigenvalues), std::move(eigenvectors)};
}
}
