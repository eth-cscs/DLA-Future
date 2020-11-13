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

#include "dlaf/auxiliary/norm/mc.h"
#include "dlaf/blas/enum_output.h"
#include "dlaf/common/assert.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/lapack/enum_output.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace auxiliary {

/// Compute the norm @p norm_type of the distributed Matrix @p A (ge/sy/he)
///
/// - With @p norm_type == lapack::Norm::Max:
///   - With @p uplo == blas::uplo::Lower
///   @pre @p A must be square matrix, A.size().rows() == A.size().cols()
///   - With @p uplo == blas::uplo::Upper
///   @note not yet implemented
///   - With @p uplo == blas::uplo::General
///   @note not yet implemented
/// - With @p norm_type = lapack::Norm::{One, Two, Inf, Fro}
/// @note not yet implemented
///
/// @pre `A.blockSize().rows() == A.blockSize().cols()`,
/// @pre @p A is distributed according to @p grid,
/// @return the norm @p norm_type of the Matrix @p A or 0 if `A.size().isEmpty()` (see LAPACK doc for
/// additional info).
template <Backend backend, Device device, class T>
dlaf::BaseType<T> norm(comm::CommunicatorGrid grid, comm::Index2D rank, lapack::Norm norm_type,
                       blas::Uplo uplo, Matrix<const T, device>& A) {
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
      DLAF_UNIMPLEMENTED(norm_type);
      return {};
    case lapack::Norm::Max:
      switch (uplo) {
        case blas::Uplo::Lower:
          return internal::Norm<backend, device, T>::max_L(grid, rank, A);
        case blas::Uplo::Upper:
        case blas::Uplo::General:
          DLAF_UNIMPLEMENTED(norm_type, uplo);
          return {};
        default:
          return {};
      }
    default:
      return {};
  }
}

}
}
