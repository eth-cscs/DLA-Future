//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/factorization/cholesky/mc/cholesky_L.h"
#include "dlaf/matrix.h"
#include "dlaf/util_matrix.h"

namespace dlaf {

template <class T>
void Factorization<Backend::MC>::cholesky(blas::Uplo uplo, Matrix<T, Device::CPU>& mat_a) {
  // Check if matrix is square
  DLAF_ASSERT_SIZE_SQUARE(mat_a);
  // Check if block matrix is square
  DLAF_ASSERT_BLOCKSIZE_SQUARE(mat_a);
  // Check if matrix is stored on local memory
  DLAF_ASSERT_LOCALMATRIX(mat_a);

  if (uplo == blas::Uplo::Lower)
    internal::mc::cholesky_L(mat_a);
  else
    throw std::runtime_error("uplo = Upper not yet implemented");
}

template <class T>
void Factorization<Backend::MC>::cholesky(comm::CommunicatorGrid grid, blas::Uplo uplo,
                                          Matrix<T, Device::CPU>& mat_a) {
  // Check if matrix is square
  DLAF_ASSERT_SIZE_SQUARE(mat_a);
  // Check if block matrix is square
  DLAF_ASSERT_BLOCKSIZE_SQUARE(mat_a);
  // Check compatibility of the communicator grid and the distribution
  DLAF_ASSERT_DISTRIBUTED_ON_GRID(grid, mat_a);

  // Method only for Lower triangular matrix
  if (uplo == blas::Uplo::Lower)
    internal::mc::cholesky_L(grid, mat_a);
  else
    throw std::runtime_error("uplo = Upper not yet implemented");
}

}
