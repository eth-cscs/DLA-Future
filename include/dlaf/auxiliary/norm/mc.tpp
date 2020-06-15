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

#include "dlaf/common/assert.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

#include "dlaf/auxiliary/norm/mc/norm_max_L.h"

namespace dlaf {

template <class T>
dlaf::BaseType<T> Auxiliary<Backend::MC>::norm(comm::CommunicatorGrid grid, comm::Index2D rank,
                                               lapack::Norm norm_type, blas::Uplo uplo,
                                               Matrix<const T, Device::CPU>& A) {
  using dlaf::matrix::equal_process_grid;

  DLAF_ASSERT(equal_process_grid(A, grid), A, grid);

  // LAPACK documentation specify that if any dimension is 0, the result is 0
  if (A.size().isEmpty())
    return {0};

  switch (norm_type) {
    case lapack::Norm::One:
    case lapack::Norm::Two:
    case lapack::Norm::Inf:
    case lapack::Norm::Fro:
      DLAF_ASSERT(false, "not yet implemented", lapack::norm2str(norm_type));
      return {};
    case lapack::Norm::Max:
      switch (uplo) {
        case blas::Uplo::Lower:
          return internal::mc::norm_max_L(grid, rank, A);
        case blas::Uplo::Upper:
        case blas::Uplo::General:
          DLAF_ASSERT(false, "not yet implemented", lapack::norm2str(norm_type), blas::uplo2str(uplo));
          return {};
        default:
          return {};
      }
    default:
      return {};
  }
}

}
