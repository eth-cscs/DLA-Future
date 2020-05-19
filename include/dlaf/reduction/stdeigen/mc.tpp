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
#include "dlaf/reduction/stdeigen/mc/stdeigen_Linv.h"
#include "dlaf/util_matrix.h"

namespace dlaf {

template <class T>
void Reduction<Backend::MC>::stdeigen(Matrix<T, Device::CPU>& mat_a,
                                      Matrix<const T, Device::CPU>& mat_l) {
  // Check if matrix A is square
  util_matrix::assertSizeSquare(mat_a, "StdeigenReduction", "mat_a");
  // Check if block matrix A is square
  util_matrix::assertBlocksizeSquare(mat_a, "StdeigenReduction", "mat_a");
  // Check if matrix A is stored on local memory
  util_matrix::assertLocalMatrix(mat_a, "StdeigenReduction", "mat_a");
  // Check if matrix L is stored on local memory
  util_matrix::assertLocalMatrix(mat_l, "StdeigenReduction", "mat_l");

  internal::mc::stdeigen(mat_a, mat_l);
}

template <class T>
void Reduction<Backend::MC>::stdeigen(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_a,
                                      Matrix<const T, Device::CPU>& mat_l) {
  // Check if matrix A is square
  util_matrix::assertSizeSquare(mat_a, "StdeigenReduction", "mat_a");
  // Check if block matrix A is square
  util_matrix::assertBlocksizeSquare(mat_a, "StdeigenReduction", "mat_a");
  // Check compatibility of the communicator grid and the distribution of matrix A
  util_matrix::assertMatrixDistributedOnGrid(grid, mat_a, "StdeigenReduction", "mat_a", "grid");
  // Check compatibility of the communicator grid and the distribution of matrix B
  util_matrix::assertMatrixDistributedOnGrid(grid, mat_l, "StdeigenReduction", "mat_l", "grid");

  //  internal::mc::stdeigen(grid, mat_a, mat_l);
  throw std::runtime_error("Distributed version not yet implemented");
}

}
