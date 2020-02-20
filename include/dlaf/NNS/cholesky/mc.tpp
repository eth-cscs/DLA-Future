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
#include "dlaf/matrix.h"
#include "dlaf/util_matrix.h"
#include "dlaf/NNS/cholesky/mc/cholesky_L.h"

namespace dlaf {
namespace NNS {

template <class T>
void NST<Execution::MC>::cholesky(blas::Uplo uplo, Matrix<T, Device::CPU>& mat_a) {
  // Check if matrix is square
  util_matrix::assertSizeSquare(mat_a, "Cholesky", "mat_a");
  // Check if block matrix is square
  util_matrix::assertBlocksizeSquare(mat_a, "Cholesky", "mat_a");
  // Check if matrix is stored on local memory
  util_matrix::assertLocalMatrix(mat_a, "Cholesky", "mat_a");

  if (uplo == blas::Uplo::Lower)
    internal::mc::cholesky_L(mat_a);
  else {
    throw std::runtime_error("uplo = Upper not yet implemented");
  }
}

template <class T>
void NST<Execution::MC>::cholesky(comm::CommunicatorGrid grid, blas::Uplo uplo, Matrix<T, Device::CPU>& mat_a) {
  // Check if matrix is square
  util_matrix::assertSizeSquare(mat_a, "Cholesky", "mat_a");
  // Check if block matrix is square
  util_matrix::assertBlocksizeSquare(mat_a, "Cholesky", "mat_a");
  // Check compatibility of the communicator grid and the distribution
  util_matrix::assertMatrixDistributedOnGrid(grid, mat_a, "Cholesky", "mat_a", "grid");

  // Method only for Lower triangular matrix
  if (uplo == blas::Uplo::Lower)
    internal::mc::cholesky_L(grid, mat_a);
  else {
    throw std::runtime_error("uplo = Upper not yet implemented");
  }
}

}
}
