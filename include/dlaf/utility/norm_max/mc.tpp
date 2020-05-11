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

#include <blas_util.hh>
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"
#include "dlaf/utility/norm_max/mc/norm_max.h"

namespace dlaf {

template <class T>
dlaf::BaseType<T> Utility<Backend::MC>::norm_max(comm::CommunicatorGrid grid, blas::Uplo uplo,
                                                 Matrix<const T, Device::CPU>& mat_a) {
  // Check if matrix is square
  DLAF_ASSERT_SIZE_SQUARE(mat_a);
  // Check if block matrix is square
  DLAF_ASSERT_BLOCKSIZE_SQUARE(mat_a);
  // Check compatibility of the communicator grid and the distribution
  DLAF_ASSERT_DISTRIBUTED_ON_GRID(grid, mat_a);

  switch (uplo) {
    case blas::Uplo::Lower:
      return internal::mc::norm_max(grid, mat_a);
    case blas::Uplo::Upper:
      throw std::runtime_error("uplo = Upper not yet implemented");
    case blas::Uplo::General:
      throw std::runtime_error("uplo = General not yet implemented");
    default:
      return {};
  }
}

}
