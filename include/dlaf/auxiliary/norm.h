//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

/// @file

#include <dlaf/auxiliary/norm/api.h>
#include <dlaf/blas/enum_output.h>
#include <dlaf/common/assert.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/lapack/enum_output.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace dlaf::auxiliary {

/// Compute the max norm of the distributed Matrix @p A (ge/sy/he)
///
/// @note @p uplo == blas::uplo::Upper not yet implemented
///
/// @pre @p A is distributed according to @p grid
/// @pre @p A has blocksize (NB x NB)
/// @pre @p A has tilesize (NB x NB)
/// @return the max norm of the Matrix @p A or 0 if `A.size().isEmpty()`
template <Backend backend, Device device, class T>
dlaf::BaseType<T> max_norm(comm::CommunicatorGrid& grid, comm::Index2D rank, blas::Uplo uplo,
                           Matrix<const T, device>& A) {
  using dlaf::matrix::equal_process_grid;
  using dlaf::matrix::single_tile_per_block;

  DLAF_ASSERT(equal_process_grid(A, grid), A, grid);
  DLAF_ASSERT(single_tile_per_block(A), A);

  // LAPACK documentation specify that if any dimension is 0, the result is 0
  if (A.size().isEmpty())
    return {0};

  switch (uplo) {
    case blas::Uplo::Lower:
      return internal::Norm<backend, device, T>::max_L(grid, rank, A);
    case blas::Uplo::Upper:
      DLAF_UNIMPLEMENTED(uplo);
      return {};
    case blas::Uplo::General:
      return internal::Norm<backend, device, T>::max_G(grid, rank, A);
    default:
      return {};
  }
}

}
