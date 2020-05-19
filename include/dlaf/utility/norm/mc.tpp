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

#include "dlaf/utility/norm/mc/norm_max_L.h"

namespace dlaf {

template <class T>
dlaf::BaseType<T> Utility<Backend::MC>::norm(comm::CommunicatorGrid grid, lapack::Norm norm_type,
                                             blas::Uplo uplo, Matrix<const T, Device::CPU>& A) {
  DLAF_ASSERT_DISTRIBUTED_ON_GRID(grid, A);

  switch (norm_type) {
    case lapack::Norm::One:
    case lapack::Norm::Two:
    case lapack::Norm::Inf:
    case lapack::Norm::Fro:
      DLAF_CHECK("", false, "not yet implemented\n", lapack::norm2str(norm_type));
    case lapack::Norm::Max:
      switch (uplo) {
        case blas::Uplo::Lower:
          return internal::mc::norm_max_L(grid, A);
        case blas::Uplo::Upper:
        case blas::Uplo::General:
          DLAF_CHECK("", false, "not yet implemented\n", lapack::norm2str(norm_type), " ", blas::uplo2str(uplo));
        default:
          return {};
      }
      break;
  }
}

}
