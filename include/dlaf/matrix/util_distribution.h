//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

/// @file
/// More details about how a matrix is distributed can be found in `misc/matrix_distribution.md`

#pragma once
#include "dlaf/common/assert.h"
#include "dlaf/types.h"

namespace dlaf {
namespace util {
namespace matrix {

/// Returns the index of the tile which contains the element with index @p element.
///
/// If the element index is local, the returned tile index is local.
/// If the element index is global, the returned tile index is global.
/// @pre 0 <= element
/// @pre 0 < block_size
inline SizeType tileFromElement(SizeType element, SizeType block_size) {
  DLAF_ASSERT_HEAVY((0 <= element));
  DLAF_ASSERT_HEAVY((0 < block_size));
  return element / block_size;
}

/// Returns the index within the tile of the element with index @p element.
///
/// The element index can be either global or local.
/// @pre 0 <= element
/// @pre 0 < block_size
inline SizeType tileElementFromElement(SizeType element, SizeType block_size) {
  DLAF_ASSERT_HEAVY((0 <= element));
  DLAF_ASSERT_HEAVY((0 < block_size));
  return element % block_size;
}

/// Returns the index of the element
/// which has index @p tile_element in the tile with index @p tile.
///
/// If the tile index is local, the returned element index is local.
/// If the tile index is global, the returned element index is global.
/// @pre 0 <= tile
/// @pre 0 <= tile_element < block_size
/// @pre 0 < block_size
inline SizeType elementFromTileAndTileElement(SizeType tile, SizeType tile_element,
                                              SizeType block_size) {
  DLAF_ASSERT_HEAVY((0 <= tile));
  DLAF_ASSERT_HEAVY((0 <= tile_element && tile_element < block_size));
  DLAF_ASSERT_HEAVY((0 < block_size));
  return tile * block_size + tile_element;
}

/// Returns the rank index of the process that stores the tiles with index @p global_tile.
///
/// @pre 0 <= global_tile
/// @pre 0 < grid_size
/// @pre 0 <= src_rank < grid_size
inline int rankGlobalTile(SizeType global_tile, int grid_size, int src_rank) {
  DLAF_ASSERT_HEAVY((0 <= global_tile));
  DLAF_ASSERT_HEAVY((0 < grid_size));
  DLAF_ASSERT_HEAVY((0 <= src_rank && src_rank < grid_size));
  return (global_tile + src_rank) % grid_size;
}

/// Returns the local tile index in process @p rank of the tile with index @p global_tile.
///
/// If the tiles with @p global_tile index is not stored by @p rank it returns -1.
/// @pre 0 <= global_tile
/// @pre 0 < grid_size
/// @pre 0 <= rank < grid_size
/// @pre 0 <= src_rank < grid_size
inline SizeType localTileFromGlobalTile(SizeType global_tile, int grid_size, int rank, int src_rank) {
  DLAF_ASSERT_HEAVY((0 <= global_tile));
  DLAF_ASSERT_HEAVY((0 < grid_size));
  DLAF_ASSERT_HEAVY((0 <= rank && rank < grid_size));
  DLAF_ASSERT_HEAVY((0 <= src_rank && src_rank < grid_size));

  if (rank == rankGlobalTile(global_tile, grid_size, src_rank))
    return global_tile / grid_size;
  else
    return -1;
}

/// Returns the local index in process @p rank of global tile
/// whose index is the smallest index larger or equal @p global_tile
/// and which is stored in process @p rank.
///
/// @pre 0 <= global_tile
/// @pre 0 < grid_size
/// @pre 0 <= rank < grid_size
/// @pre 0 <= src_rank < grid_size
inline SizeType nextLocalTileFromGlobalTile(SizeType global_tile, int grid_size, int rank,
                                            int src_rank) {
  DLAF_ASSERT_HEAVY((0 <= global_tile));
  DLAF_ASSERT_HEAVY((0 < grid_size));
  DLAF_ASSERT_HEAVY((0 <= rank && rank < grid_size));
  DLAF_ASSERT_HEAVY((0 <= src_rank && src_rank < grid_size));

  // Renumber ranks such that src_rank is 0.
  int rank_to_src = (rank + grid_size - src_rank) % grid_size;
  SizeType owner_to_src = global_tile % grid_size;

  SizeType local_tile = global_tile / grid_size;

  if (rank_to_src < owner_to_src)
    return local_tile + 1;
  else
    return local_tile;
}

/// Returns the global tile index of the tile that has index @p local_tile
/// in the process with index @p rank.
///
/// @pre 0 <= local_tile
/// @pre 0 < grid_size
/// @pre 0 <= rank < grid_size
/// @pre 0 <= src_rank < grid_size
inline SizeType globalTileFromLocalTile(SizeType local_tile, int grid_size, int rank, int src_rank) {
  DLAF_ASSERT_HEAVY((0 <= local_tile));
  DLAF_ASSERT_HEAVY((0 < grid_size));
  DLAF_ASSERT_HEAVY((0 <= rank && rank < grid_size));
  DLAF_ASSERT_HEAVY((0 <= src_rank && src_rank < grid_size));

  // Renumber ranks such that src_rank is 0.
  int rank_to_src = (rank + grid_size - src_rank) % grid_size;

  return grid_size * local_tile + rank_to_src;
}

}
}
}
