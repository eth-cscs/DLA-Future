//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/util_matrix.h"

#include "dlaf/eigensolver/reduction_to_band/mc.h"

namespace dlaf {
namespace eigensolver {

/// Reduce a local lower Hermitian matrix to symmetric band-diagonal form, with `band = blocksize + 1`.
///
/// See the related distributed version for more details.
//
/// @param mat_a on entry it contains an Hermitian matrix, on exit it is overwritten with the
/// band-diagonal result together with the elementary reflectors. Just the tiles of the lower
/// triangular part will be used.
///
/// @pre mat_a has a square size
/// @pre mat_a has a square block size
/// @pre mat_a is a local matrix
template <Backend backend, Device device, class T>
std::vector<hpx::shared_future<common::internal::vector<T>>> reductionToBand(Matrix<T, device>& mat_a) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);

  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);

  return internal::ReductionToBand<backend, device, T>::call(mat_a);
}

/// Reduce a distributed lower Hermitian matrix to symmetric band-diagonal form, with `band = blocksize + 1`.
///
/// The reduction from a lower Hermitian matrix to the band-diagonal form is performed by an orthogonal
/// similarity transformation Q, applied from left and right as in equation `Q**H . A . Q`, and whose
/// result is stored in-place in @p mat_a.
///
/// The Q matrix is a product of elementary Householder reflectors
/// `Q` = H(1) . H(2) . ... . H(n)
///
/// with `H(i) = I - tau(i) * v(i) . v(i)**H`
///
/// which are stored, together with the resulting band-diagonal matrix, in-place in the lower triangular
/// part of @p mat_a.
///
/// In particular, @p mat_a will look like this (tile representation)
///
/// B ~ ~ ~ ~ ~
/// * B ~ ~ ~ ~
/// v * B ~ ~ ~
/// v v * B ~ ~
/// v v v * B ~
/// v v v v * B
///
/// where each column of `v` is an elementary reflector without its first element (which is always equal
/// to 1), `B` are the tiles containg the band-diagonal form, while `*` tiles contain both elements
/// of the band (upper triangular diagonal included) and of the elementary reflectors (lower triangular
/// diagonal excluded).
///
/// @param grid is the CommunicatorGrid on which @p mat_a is distributed
/// @param mat_a on entry it contains an Hermitian matrix, on exit it is overwritten with the
/// band-diagonal result together with the elementary reflectors as described above. Just the tiles of
/// the lower triangular part will be used.
/// @pre mat_a has a square size
/// @pre mat_a has a square block size
/// @pre mat_a is distributed according to @p grid
template <Backend backend, Device device, class T>
std::vector<hpx::shared_future<common::internal::vector<T>>> reductionToBand(comm::CommunicatorGrid grid,
                                                                             Matrix<T, device>& mat_a) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);

  return internal::ReductionToBand<backend, device, T>::call(grid, mat_a);
}

}
}
