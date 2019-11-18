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
#include "dlaf/types.h"

namespace dlaf {
namespace util {
namespace matrix {

///
inline SizeType tileFromElement(SizeType element, SizeType tile_size){
  return element / tile_size;
}

inline SizeType tileElementFromElement(SizeType element, SizeType tile_size){
  return element % tile_size;
}

inline SizeType elementFromTileAndTileElement(SizeType tile, SizeType tile_element, SizeType tile_size){
  return tile * tile_size + tile_element;
}

inline int rankGlobalTile(SizeType global_tile, int grid_size, int src_rank) {
  return (global_tile + src_rank) % grid_size;
}

inline SizeType localTileFromGlobalTile(SizeType global_tile, int grid_size, int rank, int src_rank) {
  if (rank == rankGlobalTile(global_tile, grid_size, src_rank))
    return global_tile / grid_size;
  else
    return -1;
}

inline SizeType nextLocalTileFromGlobalTile(SizeType global_tile, int grid_size, int rank, int src_rank) {
  // Renumber ranks such that src_rank is 0.
  int rank_to_src = (rank + grid_size - src_rank) % grid_size;
  SizeType owner_to_src = global_tile % grid_size;

  SizeType local_tile = global_tile / grid_size;

  if (rank_to_src < owner_to_src)
    return local_tile + 1;
  else
    return local_tile;
}

inline SizeType globalTileFromLocalTile(SizeType local_tile, int grid_size, int rank, int src_rank) {
  // Renumber ranks such that src_rank is 0.
  int rank_to_src = (rank + grid_size - src_rank) % grid_size;

  return grid_size * local_tile + rank_to_src;
}

}
}
}
