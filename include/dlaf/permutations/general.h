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

#include <blas.hh>

#include <dlaf/common/assert.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/permutations/general/api.h>
#include <dlaf/util_matrix.h>

namespace dlaf::permutations {

/// Permutes the columns or rows of an input sub-matrix mat_in[i_begin:i_end][i_begin:i_end] into an
/// output sub-matrix mat_out[i_begin:i_end][i_begin:i_end] using the index map of permutations
/// perms[i_begin:i_end].
///
/// @param perms is the index map of permutations represented as a tiled column vector. Indices are in
///        the range [0, n) where `n` is the local size of the submatrix (i.e. the indices are local to the
///        submatrix, they are not global). Only tiles whose row tile coords are in the range
///        [i_begin,i_end) are accessed in read-only mode.
/// @pre @p perms is not distributed
/// @pre @p perms has blocksize (NB x MB)
/// @pre @p perms has tilesize (NB x MB)
///
/// @param mat_in is the input matrix. Only tiles whose both row and col tile coords are in
///        the range [i_begin,i_end) are accessed in read-only mode.
/// @pre @p mat_in has size (N x N)
/// @pre @p mat_in has blocksize (NB x NB)
/// @pre @p mat_in has tilesize (NB x NB)
///
/// @param mat_out is the output matrix. Only tiles whose both row and col tile coords are in
///        the range [i_begin,i_end) are accessed in write-only mode.
/// @pre @p mat_out has size (N x N)
/// @pre @p mat_out has blocksize (NB x NB)
/// @pre @p mat_out has tilesize (NB x NB)
template <Backend B, Device D, class T, Coord coord>
void permute(SizeType i_begin, SizeType i_end, Matrix<const SizeType, D>& perms,
             Matrix<const T, D>& mat_in, Matrix<T, D>& mat_out) {
  DLAF_ASSERT(matrix::local_matrix(perms), perms);
  DLAF_ASSERT(matrix::same_process_grid(mat_in, mat_out), mat_in, mat_out);

  // Note:
  // These are not implementation constraints, but more logic constraints. Indeed, these ensure that
  // the range [i_begin, i_end] is square in terms of elements (it would not make sense to have it square
  // in terms of number of tiles). Moreover, by requiring mat_in and mat_out matrices to have the same
  // shape, it is ensured that range [i_begin, i_end] is actually the same on both sides.
  DLAF_ASSERT(matrix::square_size(mat_in), mat_in);
  DLAF_ASSERT(matrix::square_blocksize(mat_in), mat_in);
  DLAF_ASSERT(matrix::equal_size(mat_in, mat_out), mat_in);
  DLAF_ASSERT(matrix::equal_blocksize(mat_in, mat_out), mat_in);

  DLAF_ASSERT(matrix::single_tile_per_block(perms), perms);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_in), mat_in);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_out), mat_out);

  DLAF_ASSERT(perms.block_size().rows() == mat_in.block_size().rows(), mat_in, perms);

  // Note:
  // perms is a column vector with a number of elements equal to the local part of matrix involved
  // in the permutation, i.e. [i_begin, i_end), along coord axis
  DLAF_ASSERT(perms.size().cols() == 1, perms);
  DLAF_ASSERT(perms.size().rows() == mat_in.distribution().local_size().template get<coord>(), perms,
              mat_in);

  DLAF_ASSERT(i_begin >= 0 && i_begin <= i_end, i_begin, i_end);
  DLAF_ASSERT(i_end <= mat_in.nr_tiles().rows(), i_end, perms);

  internal::Permutations<B, D, T, coord>::call(i_begin, i_end, perms, mat_in, mat_out);
}

/// Permutes the columns or rows of a distributed input sub-matrix @p
/// mat_in[i_begin:i_end][i_begin:i_end] into a distributed output sub-matrix
/// mat_out[i_begin:i_end][i_begin:i_end] using an index map of permutations
/// @p perms[i_begin:i_end] where indices are with respect to the submatrix.
/// @p i_begin is the starting global tile index and @p i_end is the end global tile index.
///
/// @param sub_task_chain orders non-blocking collective calls used internally. If @tparam coord is Coord::Col,
///        a row communicator pipeline is expected, otherwise if @tparam is Coord::Row a column communicator
///        pipeline is expected.
///
/// @param perms is the index map of permutations represented as a local tiled column vector. Indices are in
///        the range [0, n) where `n` is the global size of the submatrix (i.e. submatrix indices are used
///        instead of the full matrix indices). Only tiles whose row tile coords are in the range
///        [i_begin,i_end) are accessed in read-only mode.
/// @pre @p perms is not distributed
/// @pre @p perms has blocksize (NB x MB)
/// @pre @p perms has tilesize (NB x MB)
///
/// @param mat_in is the distributed input matrix. Only tiles whose both global row and col tile coords are in
///        the range [i_begin,i_end) are accessed in readwrite-mode.
/// @pre @p mat_in is distributed according to @p grid
/// @pre @p mat_in has size (N x N)
/// @pre @p mat_in has blocksize (NB x NB)
/// @pre @p mat_in has tilesize (NB x NB)
///
/// @param mat_out is the distributed output matrix. Only tiles whose both global row and col tile coords are in
///        the range [i_begin,i_end) are accessed in readwrite-mode.
/// @pre @p mat_out is distributed according to @p grid
/// @pre @p mat_out has size (N x N)
/// @pre @p mat_out has blocksize (NB x NB)
/// @pre @p mat_out has tilesize (NB x NB)
template <Backend B, Device D, class T, Coord coord>
void permute(
    comm::CommunicatorPipeline<comm::coord_to_communicator_type(orthogonal(coord))>& sub_task_chain,
    SizeType i_begin, SizeType i_end, Matrix<const SizeType, D>& perms, Matrix<const T, D>& mat_in,
    Matrix<T, D>& mat_out) {
  DLAF_ASSERT(matrix::local_matrix(perms), perms);
  DLAF_ASSERT(matrix::equal_process_grid(mat_in, sub_task_chain), mat_in, sub_task_chain);
  DLAF_ASSERT(matrix::equal_process_grid(mat_out, sub_task_chain), mat_out, sub_task_chain);

  // Note:
  // These are not implementation constraints, but more logic constraints. Indeed, these ensure that
  // the range [i_begin, i_end] is square in terms of elements (it would not make sense to have it square
  // in terms of number of tiles). Moreover, by requiring mat_in and mat_out matrices to have the same
  // shape, it is ensured that range [i_begin, i_end] is actually the same on both sides.
  DLAF_ASSERT(square_size(mat_in), mat_in);
  DLAF_ASSERT(matrix::square_blocksize(mat_in), mat_in);
  DLAF_ASSERT(matrix::equal_size(mat_in, mat_out), mat_in);
  DLAF_ASSERT(matrix::equal_blocksize(mat_in, mat_out), mat_in);

  DLAF_ASSERT(matrix::single_tile_per_block(perms), perms);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_in), mat_in);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_out), mat_out);

  DLAF_ASSERT(perms.size().rows() == mat_in.size().rows(), perms, mat_in);
  DLAF_ASSERT(perms.size().cols() == 1, perms);
  DLAF_ASSERT(perms.blockSize().rows() == mat_in.blockSize().rows(), mat_in, perms);

  DLAF_ASSERT(i_begin >= 0 && i_begin <= i_end, i_begin, i_end);
  DLAF_ASSERT(i_end <= perms.nrTiles().rows(), i_end, perms);

  internal::Permutations<B, D, T, coord>::call(sub_task_chain, i_begin, i_end, perms, mat_in, mat_out);
}

/// \overload permute
///
/// This overload clones the row communicator (if Coord::Col) or column communicator (if Coord::Row) of
/// @p grid internally.
template <Backend B, Device D, class T, Coord coord>
void permute(comm::CommunicatorGrid& grid, SizeType i_begin, SizeType i_end,
             Matrix<const SizeType, D>& perms, Matrix<const T, D>& mat_in, Matrix<T, D>& mat_out) {
  auto sub_task_chain(grid.communicator_pipeline<orthogonal(coord)>());
  permute<B, D, T, coord>(sub_task_chain, i_begin, i_end, perms, mat_in, mat_out);
}
}
