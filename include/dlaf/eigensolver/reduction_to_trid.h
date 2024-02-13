//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <dlaf/eigensolver/reduction_to_trid/api.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/util_matrix.h>

namespace dlaf::eigensolver::internal {

/// Reduce a local lower Hermitian matrix to symmetric tridiagonal form, with specified band_size.
///
/// @param mat_a on entry it contains an Hermitian matrix, on exit it is overwritten with the
/// tridiagonal result together with the elementary reflectors. Just the tiles of the lower
/// triangular part will be used.
///
/// @pre @p mat_a is not distributed
/// @pre @p mat_a has size (N x N)
/// @pre @p mat_a has tilesize (NB x NB)
///
/// @return the tau vector as needed by backtransformationReductionToBand
///
template <Backend B, Device D, class T>
TridiagResult1Stage<T> reduction_to_trid(Matrix<T, D>& mat_a) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_a), mat_a);

  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);

  return ReductionToTrid<B, D, T>::call(mat_a);
}

}
