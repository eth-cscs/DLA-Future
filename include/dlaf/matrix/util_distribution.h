//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

/// @file
/// More details about how a matrix is distributed can be found in `misc/matrix_distribution.md`.

#pragma once
#include <dlaf/common/assert.h>
#include <dlaf/types.h>

namespace dlaf {
namespace util {
namespace matrix {

/// Returns the index of the tile which contains the element with index @p element.
///
/// If the element index is local, the returned tile index is local.
/// If the element index is global, the returned tile index is global.
/// @pre 0 <= element,
/// @pre 0 < tile_size.
/// @pre 0 <= tile_el_offset < tile_size
inline SizeType tile_from_element(SizeType element, SizeType tile_size, SizeType tile_el_offset) {
  DLAF_ASSERT_HEAVY(0 <= element, element);
  DLAF_ASSERT_HEAVY(0 < tile_size, tile_size);
  DLAF_ASSERT_HEAVY(0 <= tile_el_offset && tile_el_offset < tile_size, tile_el_offset, tile_size);
  return (element + tile_el_offset) / tile_size;
}

/// Returns the index within the tile of the element with index @p element.
///
/// The element index can be either global or local.
/// @pre 0 <= element,
/// @pre 0 < tile_size,
/// @pre 0 <= tile_el_offset < tile_size.
inline SizeType tile_element_from_element(SizeType element, SizeType tile_size,
                                          SizeType tile_el_offset) {
  DLAF_ASSERT_HEAVY(0 <= element, element);
  DLAF_ASSERT_HEAVY(0 < tile_size, tile_size);
  DLAF_ASSERT_HEAVY(0 <= tile_el_offset && tile_el_offset < tile_size, tile_size);
  element += tile_el_offset;
  SizeType tile_element = element % tile_size;
  if (element < tile_size) {
    tile_element -= tile_el_offset;
  }
  return tile_element;
}

/// Returns the index of the element
/// which has index @p tile_element in the tile with index @p tile.
///
/// If the tile index is local, the returned element index is local.
/// If the tile index is global, the returned element index is global.
/// @pre 0 <= tile,
/// @pre 0 <= tile_element < tile_size,
/// @pre 0 < tile_size,
/// @pre 0 <= tile_el_offset < tile_size.
inline SizeType element_from_tile_and_tile_element(SizeType tile, SizeType tile_element,
                                                   SizeType tile_size, SizeType tile_el_offset) {
  DLAF_ASSERT_HEAVY(0 <= tile, tile);
  DLAF_ASSERT_HEAVY(0 <= tile_el_offset && tile_el_offset < tile_size, tile_el_offset, tile_size);
  DLAF_ASSERT_HEAVY(0 <= tile_element && tile_element < tile_size &&
                        (tile > 0 || tile_element < (tile_size - tile_el_offset)),
                    tile, tile_element, tile_size, tile_el_offset);
  DLAF_ASSERT_HEAVY(0 < tile_size, tile_size);
  return tile * tile_size + tile_element - (tile > 0 ? tile_el_offset : 0);
}

/// Returns the rank index of the process that stores the tiles with index @p global_tile.
///
/// @pre 0 <= global_tile,
/// @pre 0 < tiles_per_block,
/// @pre 0 < grid_size,
/// @pre 0 <= src_rank < grid_size,
/// @pre 0 <= tile_offset < tiles_per_block.
inline int rank_global_tile(SizeType global_tile, SizeType tiles_per_block, int grid_size, int src_rank,
                            SizeType tile_offset) {
  DLAF_ASSERT_HEAVY(0 <= global_tile, global_tile);
  DLAF_ASSERT_HEAVY(0 < tiles_per_block, tiles_per_block);
  DLAF_ASSERT_HEAVY(0 < grid_size, grid_size);
  DLAF_ASSERT_HEAVY(0 <= src_rank && src_rank < grid_size, src_rank, grid_size);
  DLAF_ASSERT_HEAVY(0 <= tile_offset && tile_offset < tiles_per_block, tile_offset, tiles_per_block);

  SizeType global_block = (global_tile + tile_offset) / tiles_per_block;
  return (global_block + src_rank) % grid_size;
}

/// Returns the local tile index in process @p rank of the tile with index @p global_tile.
///
/// If the tiles with @p global_tile index is not stored by @p rank it returns -1.
/// @pre 0 <= global_tile,
/// @pre 0 < tiles_per_block,
/// @pre 0 < grid_size,
/// @pre 0 <= rank < grid_size,
/// @pre 0 <= src_rank < grid_size,
/// @pre 0 <= tile_offset < tiles_per_block.
inline SizeType local_tile_from_global_tile(SizeType global_tile, SizeType tiles_per_block,
                                            int grid_size, int rank, int src_rank,
                                            SizeType tile_offset) {
  DLAF_ASSERT_HEAVY(0 <= global_tile, global_tile);
  DLAF_ASSERT_HEAVY(0 < tiles_per_block, tiles_per_block);
  DLAF_ASSERT_HEAVY(0 < grid_size, grid_size);
  DLAF_ASSERT_HEAVY(0 <= rank && rank < grid_size, rank, grid_size);
  DLAF_ASSERT_HEAVY(0 <= src_rank && src_rank < grid_size, src_rank, grid_size);
  DLAF_ASSERT_HEAVY(0 <= tile_offset && tile_offset < tiles_per_block, tile_offset, tiles_per_block);

  if (rank == rank_global_tile(global_tile, tiles_per_block, grid_size, src_rank, tile_offset)) {
    global_tile += tile_offset;

    SizeType local_block = global_tile / tiles_per_block / grid_size;

    // tile_offset only affects the source rank
    bool may_have_partial_first_block = rank == src_rank;

    return local_block * tiles_per_block + global_tile % tiles_per_block -
           (may_have_partial_first_block ? tile_offset : 0);
  }
  else
    return -1;
}

/// Returns the local index in process @p rank of global tile
/// whose index is the smallest index larger or equal @p global_tile
/// and which is stored in process @p rank.
///
/// @pre 0 <= global_tile,
/// @pre 0 < tiles_per_block,
/// @pre 0 < grid_size,
/// @pre 0 <= rank < grid_size,
/// @pre 0 <= src_rank < grid_size,
/// @pre 0 <= tile_offset < tiles_per_block.
inline SizeType next_local_tile_from_global_tile(SizeType global_tile, SizeType tiles_per_block,
                                                 int grid_size, int rank, int src_rank,
                                                 SizeType tile_offset) {
  DLAF_ASSERT_HEAVY(0 <= global_tile, global_tile);
  DLAF_ASSERT_HEAVY(0 < tiles_per_block, tiles_per_block);
  DLAF_ASSERT_HEAVY(0 < grid_size, grid_size);
  DLAF_ASSERT_HEAVY(0 <= rank && rank < grid_size, rank, grid_size);
  DLAF_ASSERT_HEAVY(0 <= src_rank && src_rank < grid_size, src_rank, grid_size);
  DLAF_ASSERT_HEAVY(0 <= tile_offset && tile_offset < tiles_per_block, tile_offset, tiles_per_block);

  int rank_to_src = (rank + grid_size - src_rank) % grid_size;
  global_tile += tile_offset;
  SizeType global_block = global_tile / tiles_per_block;
  SizeType owner_to_src = global_block % grid_size;
  SizeType local_block = global_block / grid_size;

  // If there's a tile offset it affects only the source rank block. All other
  // blocks are whole.
  bool may_have_partial_first_block = rank == src_rank;

  if (rank_to_src == owner_to_src)
    return local_block * tiles_per_block + global_tile % tiles_per_block -
           (may_have_partial_first_block ? tile_offset : 0);

  if (rank_to_src < owner_to_src)
    ++local_block;

  return local_block * tiles_per_block - (may_have_partial_first_block ? tile_offset : 0);
}

/// Returns the global tile index of the tile that has index @p local_tile
/// in the process with index @p rank.
///
/// @pre 0 <= local_tile,
/// @pre 0 < tiles_per_block,
/// @pre 0 < grid_size,
/// @pre 0 <= rank < grid_size,
/// @pre 0 <= src_rank < grid_size,
/// @pre 0 <= tile_offset < tiles_per_block.
inline SizeType global_tile_from_local_tile(SizeType local_tile, SizeType tiles_per_block, int grid_size,
                                            int rank, int src_rank, SizeType tile_offset) {
  DLAF_ASSERT_HEAVY(0 <= local_tile, local_tile);
  DLAF_ASSERT_HEAVY(0 < tiles_per_block, tiles_per_block);
  DLAF_ASSERT_HEAVY(0 < grid_size, grid_size);
  DLAF_ASSERT_HEAVY(0 <= rank && rank < grid_size, rank, grid_size);
  DLAF_ASSERT_HEAVY(0 <= src_rank && src_rank < grid_size, src_rank, grid_size);
  DLAF_ASSERT_HEAVY(0 <= tile_offset && tile_offset < tiles_per_block, tile_offset, tiles_per_block);

  bool may_have_partial_first_block = rank == src_rank;
  if (may_have_partial_first_block) {
    local_tile += tile_offset;
  }
  // Renumber ranks such that src_rank is 0.
  int rank_to_src = (rank + grid_size - src_rank) % grid_size;
  SizeType local_block = local_tile / tiles_per_block;

  return (grid_size * local_block + rank_to_src) * tiles_per_block + local_tile % tiles_per_block -
         tile_offset;
}

}
}
}
