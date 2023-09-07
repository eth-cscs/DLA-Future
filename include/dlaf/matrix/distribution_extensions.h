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

#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/util_distribution.h>
#include <dlaf/types.h>

namespace dlaf::matrix::internal::distribution {
/*
// TODO
template <Coord rc>
SizeType global_element_from_local_element_and_rank(comm::IndexT_MPI rank,
                                              const SizeType local_element) const noexcept {
  const auto local_tile = local_element / tile_size_.get<rc>();
  const auto el_tl = local_element % tile_size_.get<rc>();
  const SizeType global_tile =
      util::matrix::global_tile_from_local_tile(local_tile, tiles_per_block<rc>(), grid_size_.get<rc>(),
                                            rank, source_rank_index_.get<rc>(),
                                            global_tile_offset<rc>());
  return global_element_from_global_tile_and_tile_element<rc>(global_tile, el_tl);
}
*/

///////////////////////////
// helpers for submatrix //
///////////////////////////

/// Checks if sub-distribution is compatible with distribution.
///
/// Compatible means that the block size, tile size, rank index, and grid size are equal.
/// Sub-distribution means that the source rank index of the sub-distribution is the rank index
/// of the tile at sub_offset in the current distribution. Additionally, the size and offset of
/// the sub-distribution must be within the size of the current distribution.
inline bool is_compatible_sub_distribution(const Distribution& distribution,
                                           const GlobalElementIndex& sub_offset,
                                           const Distribution& sub_distribution) noexcept {
  const bool compatible_grid = distribution.block_size() == sub_distribution.block_size() &&
                               distribution.tile_size() == sub_distribution.tile_size() &&
                               distribution.rank_index() == sub_distribution.rank_index() &&
                               distribution.grid_size() == sub_distribution.grid_size();
  const bool compatible_source_rank_index =
      distribution.rank_global_tile(distribution.global_tile_index(sub_offset)) ==
      sub_distribution.source_rank_index();
  const bool compatible_size =
      sub_offset.row() + sub_distribution.size().rows() <= distribution.size().rows() &&
      sub_offset.col() + sub_distribution.size().cols() <= distribution.size().cols();
  return compatible_grid && compatible_source_rank_index && compatible_size;
}

/// Returns the tile index in the current distribution corresponding to a tile index @p sub_index in a
/// sub-distribution (defined by @p sub_offset and @p sub_distribution)
inline GlobalTileIndex global_tile_index_from_sub_distribution(
    const Distribution& distribution, const GlobalElementIndex& sub_offset,
    const Distribution& sub_distribution, const GlobalTileIndex& sub_index) noexcept {
  DLAF_ASSERT(sub_index.isIn(sub_distribution.nrTiles()), sub_index, sub_distribution.nrTiles());
  DLAF_ASSERT(is_compatible_sub_distribution(distribution, sub_offset, sub_distribution), "");
  const GlobalTileIndex tile_offset = distribution.globalTileIndex(sub_offset);
  return tile_offset + common::sizeFromOrigin(sub_index);
}

/// Returns the element offset within the tile in the current distribution corresponding to a tile
/// index @p sub_index in a sub-distribution (defined by @p sub_offset and @p sub_distribution)
inline TileElementIndex tile_element_offset_from_sub_distribution(
    const Distribution& distribution, const GlobalElementIndex& sub_offset,
    const Distribution& sub_distribution, const GlobalTileIndex& sub_index) noexcept {
  DLAF_ASSERT(sub_index.isIn(sub_distribution.nrTiles()), sub_index, sub_distribution.nrTiles());
  DLAF_ASSERT(is_compatible_sub_distribution(distribution, sub_offset, sub_distribution), "");
  return {
      sub_index.row() == 0 ? distribution.tile_element_from_global_element<Coord::Row>(sub_offset.row())
                           : 0,
      sub_index.col() == 0 ? distribution.tile_element_from_global_element<Coord::Col>(sub_offset.col())
                           : 0,
  };
}

/////////////////////////////////////////
// helpers for tridiagonal eigensolver //
/////////////////////////////////////////

/// Returns the distance from the global index @p i_gl to the tile adjacent the one containing @p i_gl
/// along @p rc coordinate.
template <Coord rc>
SizeType distance_to_adjacent_tile(const Distribution& dist, SizeType global_element) noexcept {
  const SizeType global_tile = dist.global_tile_from_global_element<rc>(global_element);
  const SizeType tile_element = dist.tile_element_from_global_element<rc>(global_element);
  return dist.size_of_tile<rc>(global_tile) - tile_element;
}

/// Returns a local linear column-major index of the tile that contains @p ij_local
inline SizeType local_tile_linear_index(const Distribution& dist, LocalTileIndex ij_local) noexcept {
  return ij_local.row() + ij_local.col() * dist.local_nr_tiles().rows();
}

/// Returns a global linear column-major index of the tile that contains @p ij
inline SizeType global_tile_linear_index(const Distribution& dist, GlobalElementIndex ij) noexcept {
  GlobalTileIndex tile_i = dist.global_tile_index(ij);
  return tile_i.row() + tile_i.col() * dist.nr_tiles().rows();
}

/// Returns the global element distance between tiles along the @p rc coordinate
template <Coord rc>
SizeType global_tile_element_distance(const Distribution& dist, SizeType i_begin,
                                      SizeType i_end) noexcept {
  DLAF_ASSERT_HEAVY(i_begin <= i_end, i_begin, i_end);

  const SizeType el_begin = dist.global_element_from_global_tile_and_tile_element<rc>(i_begin, 0);

  if (i_end == dist.nr_tiles().get<rc>())
    return dist.size().get<rc>() - el_begin;

  return dist.global_element_from_global_tile_and_tile_element<rc>(i_end, 0) - el_begin;
}

/// Returns the local element size of the region between the local tile indices @p i_loc_begin and @p
/// i_loc_end along the @p rc coordinate
template <Coord rc>
SizeType local_element_distance_from_local_tile(const Distribution& dist, SizeType i_loc_begin,
                                                SizeType i_loc_end) noexcept {
  DLAF_ASSERT_HEAVY(i_loc_begin <= i_loc_end, i_loc_begin, i_loc_end);
  DLAF_ASSERT_HEAVY(0 <= i_loc_begin && i_loc_end <= dist.local_nr_tiles().get<rc>(), i_loc_begin,
                    i_loc_end, dist.local_nr_tiles().get<rc>());

  SizeType i_loc_last = i_loc_end - 1;
  if (i_loc_begin > i_loc_last)
    return 0;

  SizeType i_el_begin = distribution.local_element_fromlocal_tile_and_tile_element(i_loc_begin, 0);
  SizeType i_el_end = util::matrix::elementFromTileAndTileElement(
                          i_loc_last, tileSize<rc>(globalTileFromLocalTile<rc>(i_loc_last)) - 1) +
                      1;
  return i_el_end - i_el_begin;
}

/// Returns the local element size of the region between global tile indices @p i_begin and @p i_end
/// along the @p rc coordinate
template <Coord rc>
SizeType local_element_distance_from_global_tile(const Distribution& dist, SizeType i_begin,
                                                 SizeType i_end) noexcept {
  DLAF_ASSERT_HEAVY(i_begin <= i_end, i_begin, i_end);
  DLAF_ASSERT_HEAVY(0 <= i_begin && i_end <= dist.nr_tiles().get<rc>(), i_begin, i_end,
                    dist.nr_tiles().get<rc>());

  SizeType i_loc_begin = dist.next_local_tile_from_global_tile<rc>(i_begin);
  SizeType i_loc_end = dist.next_local_tile_from_global_tile<rc>(i_end);
  return local_element_distance_from_local_tile<rc>(dist, i_loc_begin, i_loc_end);
}

/// This overload implements the 2D version of the function.
inline LocalElementSize local_element_distance_from_local_tile(const Distribution& dist,
                                                               LocalTileIndex begin,
                                                               LocalTileIndex end) noexcept {
  return {local_element_distance_from_local_tile<Coord::Row>(dist, begin.row(), end.row()),
          local_element_distance_from_local_tile<Coord::Col>(dist, begin.col(), end.col())};
}
}
