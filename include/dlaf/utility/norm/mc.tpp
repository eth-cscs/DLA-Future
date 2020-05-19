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
      throw std::runtime_error("lapack::Norm::One not yet implemented");
    case lapack::Norm::Two:
      throw std::runtime_error("lapack::Norm::Two not yet implemented");
    case lapack::Norm::Inf:
      throw std::runtime_error("lapack::Norm::Inf not yet implemented");
    case lapack::Norm::Fro:
      throw std::runtime_error("lapack::Norm::Fro not yet implemented");
    case lapack::Norm::Max:
      switch (uplo) {
        case blas::Uplo::Lower:
          return internal::mc::norm_max_L(grid, A);
        case blas::Uplo::Upper:
          throw std::runtime_error("uplo = Upper not yet implemented");
        case blas::Uplo::General:
          throw std::runtime_error("uplo = General not yet implemented");
        default:
          return {};
      }
      break;
  }
}
}
