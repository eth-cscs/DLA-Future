//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

/// @file

#include <blas.hh>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/inverse/triangular/api.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace dlaf {

/// Invert a non-singular triangular matrix in-place.
///
/// @param uplo specifies if the elements of the Hermitian matrix to be referenced are the elements in
/// the lower or upper triangular part,
/// @param diag specifies if the matrix A is assumed to be unit triangular (\a Unit) or not (\a NonUnit),
/// @param mat_a on entry it contains the triangular matrix A, on exit the matrix elements
/// are overwritten with the elements of the inverse. Only the tiles of the matrix
/// which contain the upper or the lower triangular part (depending on the value of uplo),
/// are accessed and modified.
///
/// @pre @p mat_a is not distributed
/// @pre @p mat_a has size (N x N)
/// @pre @p mat_a has tile size (NB x NB)
template <Backend backend, Device device, class T>
void triangular_inverse(blas::Uplo uplo, blas::Diag diag, Matrix<T, device>& mat_a) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_tile_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);

  if (uplo == blas::Uplo::Lower)
    inverse::internal::Triangular<backend, device, T>::call_L(diag, mat_a);
  // else
  //   inverse::internal::Triangular<backend, device, T>::call_U(diag, mat_a);
}

/// Invert a non-singular triangular matrix in-place.
///
/// @param grid is the communicator grid on which the matrix A has been distributed,
/// @param uplo specifies if the elements of the Hermitian matrix to be referenced are the elements in
/// the lower or upper triangular part,
/// @param diag specifies if the matrix A is assumed to be unit triangular (\a Unit) or not (\a NonUnit),
/// @param mat_a on entry it contains the triangular matrix A, on exit the matrix elements
/// are overwritten with the elements of the inverse. Only the tiles of the matrix
/// which contain the upper or the lower triangular part (depending on the value of uplo),
/// are accessed and modified.
///
/// @pre @p mat_a is distributed according to @p grid
/// @pre @p mat_a has size (N x N)
/// @pre @p mat_a has tile size (NB x NB)
template <Backend backend, Device device, class T>
void triangular_inverse(comm::CommunicatorGrid& grid, blas::Uplo uplo, blas::Diag diag,
                        Matrix<T, device>& mat_a) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_tile_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);

  // if (uplo == blas::Uplo::Lower)
  //   inverse::internal::Triangular<backend, device, T>::call_L(grid, diag, mat_a);
  // else
  //   inverse::internal::Triangular<backend, device, T>::call_U(grid, diag, mat_a);
}

}
