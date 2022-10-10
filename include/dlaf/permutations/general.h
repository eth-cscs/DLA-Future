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

#include <blas.hh>

#include "dlaf/common/assert.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/permutations/general/api.h"
#include "dlaf/util_matrix.h"

namespace dlaf::permutations {

/// Permutes the columns or rows of an input sub-matrix mat_in[i_begin:i_end][i_begin:i_end] into an
/// output sub-matrix mat_out[i_begin:i_end][i_begin:i_end] using the index map of permutations
/// perms[i_begin:i_end].
///
/// @param perms is the index map of permutations represented as a tiled column vector. Indices are in
///        the range [0, n) where `n` is the size of the submatrix (i.e. the indices are local to the
///        submatrix, they are not global). Only tiles whose row tile coords are in the closed range
///        [i_begin,i_end] are accessed in read-only mode.
/// @param mat_in is the input matrix. Only tiles whose both row and col tile coords are in
///        the closed range [i_begin,i_end] are accessed in read-only mode.
/// @param mat_out is the output matrix. Only tiles whose both row and col tile coords are in
///        the closed range [i_begin,i_end] are accessed in write-only mode.
///
template <Backend B, Device D, class T, Coord coord>
void permute(SizeType i_begin, SizeType i_end, Matrix<const SizeType, D>& perms, Matrix<T, D>& mat_in,
             Matrix<T, D>& mat_out) {
  const matrix::Distribution& distr_perms = perms.distribution();
  const matrix::Distribution& distr_in = mat_in.distribution();
  const matrix::Distribution& distr_out = mat_out.distribution();

  DLAF_ASSERT(matrix::local_matrix(perms), perms);
  DLAF_ASSERT(matrix::local_matrix(mat_in), mat_in);
  DLAF_ASSERT(matrix::local_matrix(mat_out), mat_out);

  DLAF_ASSERT(i_begin >= 0 && i_begin <= i_end, i_begin, i_end);

  DLAF_ASSERT(i_end < distr_perms.nrTiles().rows(), i_end, perms);
  DLAF_ASSERT(i_end < distr_in.nrTiles().rows() && i_end < distr_in.nrTiles().cols(), i_end, mat_in);
  DLAF_ASSERT(i_end < distr_out.nrTiles().rows() && i_end < distr_out.nrTiles().cols(), i_end, mat_out);

  DLAF_ASSERT(distr_perms.size().cols() == 1, perms);

  DLAF_ASSERT(matrix::equal_blocksize(mat_in, mat_out), mat_in, mat_out);
  DLAF_ASSERT(distr_in.blockSize().get<coord>() == distr_perms.blockSize().rows(), mat_in, perms);

  internal::Permutations<B, D, T, coord>::call(i_begin, i_end, perms, mat_in, mat_out);
}

template <Backend B, Device D, class T, Coord coord>
void permute(comm::CommunicatorGrid grid, SizeType i_begin, SizeType i_end,
             Matrix<const SizeType, D>& perms, Matrix<T, D>& mat_in, Matrix<T, D>& mat_out) {
  const matrix::Distribution& distr_perms = perms.distribution();
  const matrix::Distribution& distr_in = mat_in.distribution();

  DLAF_ASSERT(matrix::square_size(mat_in), mat_in);
  DLAF_ASSERT(matrix::square_blocksize(mat_in), mat_in);
  DLAF_ASSERT(matrix::equal_blocksize(mat_in, mat_out), mat_in, mat_out);
  DLAF_ASSERT(matrix::equal_size(mat_in, mat_out), mat_in, mat_out);

  DLAF_ASSERT(matrix::local_matrix(perms), perms);
  DLAF_ASSERT(distr_perms.size().cols() == 1, perms);
  DLAF_ASSERT(distr_in.blockSize().rows() == distr_perms.blockSize().rows(), mat_in, perms);

  internal::Permutations<B, D, T, coord>::call(grid, i_begin, i_end, perms, mat_in, mat_out);
}

}
