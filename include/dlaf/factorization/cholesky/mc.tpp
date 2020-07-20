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
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);

  if (uplo == blas::Uplo::Lower)
    internal::mc::cholesky_L(mat_a);
  else {
    std::cout << "uplo = Upper not yet implemented" << std::endl;
    std::abort();
  }
}

template <class T>
void Factorization<Backend::MC>::cholesky(comm::CommunicatorGrid grid, blas::Uplo uplo,
                                          Matrix<T, Device::CPU>& mat_a) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);

  // Method only for Lower triangular matrix
  if (uplo == blas::Uplo::Lower)
    internal::mc::cholesky_L(grid, mat_a);
  else {
    std::cout << "uplo = Upper not yet implemented" << std::endl;
    std::abort();
  }
}

}
