//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file
/// Utilities to check if a matrix is allocated correctly according to the given AllocationLayout parameter.
///
/// These functions are included in DLAF as they can be used by the users to test custom defined layouts.
/// Note: These functions are blocking and meant only for testing purpose. Do not use them in applications.

#include <pika/execution.hpp>

#include <dlaf/common/range2d.h>
#include <dlaf/matrix/allocation.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace dlaf::matrix::test {

namespace internal {
template <class MatrixLike>
auto* sync_tile_ptr(MatrixLike& mat, const LocalTileIndex& ij) {
  namespace tt = pika::this_thread::experimental;
  return tt::sync_wait(mat.read(ij)).get().ptr();
}

template <class MatrixLike>
SizeType sync_tile_ld(MatrixLike& mat, const LocalTileIndex& ij) {
  namespace tt = pika::this_thread::experimental;
  return tt::sync_wait(mat.read(ij)).get().ld();
}
}

template <class MatrixLike>
bool is_allocated_as_col_major(MatrixLike& mat) {
  if (mat.allocation() != AllocationLayout::ColMajor)
    return false;

  Distribution dist = mat.distribution();

  if (dist.local_nr_tiles().isEmpty())
    return true;

  auto* ptr_00 = internal::sync_tile_ptr(mat, {0, 0});
  SizeType ld_00 = internal::sync_tile_ld(mat, {0, 0});

  for (auto& ij : iterate_range2d(dist.local_nr_tiles())) {
    SizeType i_el_local = dist.local_element_from_local_tile_and_tile_element<Coord::Row>(ij.row(), 0);
    SizeType j_el_local = dist.local_element_from_local_tile_and_tile_element<Coord::Col>(ij.col(), 0);
    SizeType offset = i_el_local + ld_00 * j_el_local;
    auto* ptr_ij = internal::sync_tile_ptr(mat, ij);
    SizeType ld_ij = internal::sync_tile_ld(mat, ij);
    if (ptr_ij - ptr_00 != offset || ld_ij != ld_00)
      return false;
  }
  return true;
}

template <class MatrixLike>
bool is_allocated_as_blocks(MatrixLike& mat) {
  if (mat.allocation() != AllocationLayout::Blocks)
    return false;

  const Distribution& dist = mat.distribution();
  Distribution helper_dist = dlaf::matrix::internal::create_single_tile_per_block_distribution(dist);
  LocalBlockSize local_nr_blocks = helper_dist.local_nr_blocks();

  if (dist.local_nr_tiles().isEmpty())
    return true;

  for (SizeType j_bl = 0, j = 0; j_bl < local_nr_blocks.cols(); ++j_bl) {
    // When computing the number of tiles in a block, special care is needed when only a single block is
    // present, as it can contain two incomplete tiles.
    SizeType nb = helper_dist.local_tile_size_of<Coord::Col>(j_bl);
    SizeType tiles_in_block_cols = nb / dist.tile_size().cols();
    if (local_nr_blocks.cols() == 1)
      tiles_in_block_cols = dist.local_nr_tiles().cols();

    for (SizeType i_bl = 0, i = 0; i_bl < local_nr_blocks.rows(); ++i_bl) {
      // When computing the number of tiles in a block, special care is needed when only a single block
      // is present, as it can contain two incomplete tiles.
      SizeType mb = helper_dist.local_tile_size_of<Coord::Row>(i_bl);
      SizeType tiles_in_block_rows = mb / dist.tile_size().rows();
      if (local_nr_blocks.rows() == 1)
        tiles_in_block_rows = dist.local_nr_tiles().rows();

      // pointer and ld of the first element of the block.
      auto* ptr_bl = internal::sync_tile_ptr(mat, {i, j});
      SizeType ld_bl = internal::sync_tile_ld(mat, {i, j});
      SizeType i_el_local_bl = dist.local_element_from_local_tile_and_tile_element<Coord::Row>(i, 0);
      SizeType j_el_local_bl = dist.local_element_from_local_tile_and_tile_element<Coord::Col>(j, 0);

      for (SizeType j_tl_bl = 0; j_tl_bl < tiles_in_block_cols; ++j_tl_bl) {
        for (SizeType i_tl_bl = 0; i_tl_bl < tiles_in_block_rows; ++i_tl_bl) {
          LocalTileIndex ij{i + i_tl_bl, j + j_tl_bl};
          SizeType i_el_local =
              dist.local_element_from_local_tile_and_tile_element<Coord::Row>(ij.row(), 0);
          SizeType j_el_local =
              dist.local_element_from_local_tile_and_tile_element<Coord::Col>(ij.col(), 0);
          SizeType offset = (i_el_local - i_el_local_bl) + ld_bl * (j_el_local - j_el_local_bl);
          auto* ptr_ij = internal::sync_tile_ptr(mat, ij);
          SizeType ld_ij = internal::sync_tile_ld(mat, ij);

          if (ptr_ij - ptr_bl != offset || ld_ij != ld_bl)
            return false;
        }
      }
      i += tiles_in_block_rows;
    }
    j += tiles_in_block_cols;
  }

  return true;
}

template <class MatrixLike>
bool is_allocated_as_tiles(MatrixLike& mat) {
  return mat.allocation() == AllocationLayout::Tiles;
}

template <class MatrixLike>
bool is_allocated_as(MatrixLike& mat, AllocationLayout alloc) {
  switch (alloc) {
    case AllocationLayout::ColMajor:
      return is_allocated_as_col_major(mat);
    case AllocationLayout::Blocks:
      return is_allocated_as_blocks(mat);
    case AllocationLayout::Tiles:
      return is_allocated_as_tiles(mat);
  }
  return false;
}
}
