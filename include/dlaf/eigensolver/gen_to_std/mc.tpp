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
#include "dlaf/eigensolver/gen_to_std/mc/gen_to_std_L.h"
#include "dlaf/matrix.h"
#include "dlaf/util_matrix.h"

namespace dlaf {

template <class T>
void Eigensolver<Backend::MC>::genToStd(blas::Uplo uplo, Matrix<T, Device::CPU>& mat_a,
                                        Matrix<T, Device::CPU>& mat_l) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_size(mat_l), mat_l);
  DLAF_ASSERT(matrix::square_blocksize(mat_l), mat_l);
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);
  DLAF_ASSERT(matrix::local_matrix(mat_l), mat_l);

  if (uplo == blas::Uplo::Lower)
    internal::mc::genToStd_L(mat_a, mat_l);
  else {
    std::cout << "uplo = Upper not yet implemented" << std::endl;
    std::abort();
  }
}

}
