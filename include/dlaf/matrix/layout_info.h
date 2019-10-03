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
#include "dlaf/matrix/index.h"
#include "dlaf/util_math.h"

namespace dlaf {
namespace matrix {
class LayoutInfo {
public:
  LayoutInfo(GlobalElementSize size, TileElementSize block_size, SizeType tile_ld,
             std::size_t tile_offset_row, std::size_t tile_offset_col);

  std::size_t tileOffset(LocalTileIndex index) const {
    // TODO checks
    return index.row() * tile_offset_row_ + index.col() * tile_offset_col_;
  }

  std::size_t minMemSize() const noexcept {
    if (size_.rows() == 0 || size_.cols() == 0) {
      return 0;
    }

    SizeType last_rows = size_.rows() - block_size_.rows() * (nr_tiles_.rows() - 1);
    SizeType last_cols = size_.cols() - block_size_.cols() * (nr_tiles_.cols() - 1);

    return tileOffset({nr_tiles_.rows() - 1, nr_tiles_.cols() - 1}) +
           static_cast<std::size_t>(ld_tile_) * static_cast<std::size_t>(last_cols - 1) + last_rows;
  }

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
  std::size_t minTileSize(TileElementSize size) const noexcept;

  GlobalElementSize size_;
  LocalTileSize nr_tiles_;
  TileElementSize block_size_;

  SizeType ld_tile_;
  std::size_t tile_offset_row_;
  std::size_t tile_offset_col_;
};

inline LayoutInfo ColMajorLayout(GlobalElementSize size, TileElementSize block_size, SizeType ld) {
  using util::size_t::mul;
  return LayoutInfo(size, block_size, ld, static_cast<std::size_t>(block_size.rows()),
                    mul(block_size.cols(), ld));
}

inline LayoutInfo TileLayout(GlobalElementSize size, TileElementSize block_size, SizeType ld_tile,
                             SizeType tiles_per_col) {
  using util::size_t::mul;
  std::size_t tile_size = mul(ld_tile, block_size.cols());
  return LayoutInfo(size, block_size, ld_tile, tile_size, mul(tile_size, tiles_per_col));
}
}
}
