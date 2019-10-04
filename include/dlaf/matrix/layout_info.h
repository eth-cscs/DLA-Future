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
#include <cassert>
#include "dlaf/matrix/index.h"
#include "dlaf/util_math.h"

namespace dlaf {
namespace matrix {
class LayoutInfo {
public:
  LayoutInfo(GlobalElementSize size, TileElementSize block_size, SizeType tile_ld,
             std::size_t tile_offset_row, std::size_t tile_offset_col);

  /// @brief Returns the position of the first element of the @p index tile.
  /// @pre 0 < @p index.row() < nrTiles().rows()
  /// @pre 0 < @p index.col() < nrTiles().cols()
  std::size_t tileOffset(const LocalTileIndex index) const {
    assert(index.isValid() && index.isIn(nr_tiles_));
    return index.row() * tile_offset_row_ + index.col() * tile_offset_col_;
  }

  /// @brief Returns the minimum number of elements that are needed to fit a matrix with the given layout.
  std::size_t minMemSize() const noexcept;

  GlobalElementSize size() const noexcept {
    return size_;
  }

  LocalTileSize nrTiles() const noexcept {
    return nr_tiles_;
  }

  TileElementSize blockSize() const noexcept {
    return block_size_;
  }

  SizeType ldTile() const noexcept {
    return ld_tile_;
  }

private:
  /// @brief Returns the minimum number of elements that are needed to fit a tile with the given size.
  std::size_t minTileSize(TileElementSize size) const noexcept;

  GlobalElementSize size_;
  LocalTileSize nr_tiles_;
  TileElementSize block_size_;

  SizeType ld_tile_;
  std::size_t tile_offset_row_;
  std::size_t tile_offset_col_;
};

/// @brief Returns LayoutInfo for a column major matrix.
inline LayoutInfo ColMajorLayout(GlobalElementSize size, TileElementSize block_size, SizeType ld) {
  using util::size_t::mul;
  return LayoutInfo(size, block_size, ld, static_cast<std::size_t>(block_size.rows()),
                    mul(block_size.cols(), ld));
}

/// @brief Returns LayoutInfo for a matrix which use the tile layout.
inline LayoutInfo TileLayout(GlobalElementSize size, TileElementSize block_size, SizeType ld_tile,
                             SizeType tiles_per_col) {
  using util::size_t::mul;
  std::size_t tile_size = mul(ld_tile, block_size.cols());
  return LayoutInfo(size, block_size, ld_tile, tile_size, mul(tile_size, tiles_per_col));
}
}
}
