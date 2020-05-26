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
#include "dlaf/eigensolver/gen_to_std/mc/gen_to_std_L.h"
#include "dlaf/util_matrix.h"

namespace dlaf {

template <class T>
void Eigensolver<Backend::MC>::genToStd(Matrix<T, Device::CPU>& mat_a,
                                      Matrix<const T, Device::CPU>& mat_l) {
  // Check if matrix A is square
  util_matrix::assertSizeSquare(mat_a, "genToStd", "mat_a");
  // Check if block matrix A is square
  util_matrix::assertBlocksizeSquare(mat_a, "genToStd", "mat_a");
  // Check if matrix A is stored on local memory
  util_matrix::assertLocalMatrix(mat_a, "genToStd", "mat_a");
  // Check if matrix L is stored on local memory
  util_matrix::assertLocalMatrix(mat_l, "genToStd", "mat_l");

  internal::mc::genToStd(mat_a, mat_l);
}

}
