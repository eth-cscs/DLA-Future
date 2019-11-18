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
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
#include "dlaf/util_math.h"

namespace dlaf {
namespace matrix {
class LayoutInfo {
public:
  LayoutInfo(const LocalElementSize& size, const TileElementSize& block_size, SizeType tile_ld,
             std::size_t tile_offset_row, std::size_t tile_offset_col);

  bool operator==(const LayoutInfo& rhs) const noexcept {
    return size_ == rhs.size_ && nr_tiles_ == rhs.nr_tiles_ && block_size_ == rhs.block_size_ &&
           ld_tile_ == rhs.ld_tile_ && tile_offset_row_ == rhs.tile_offset_row_ &&
           tile_offset_col_ == rhs.tile_offset_col_;
  }

  bool operator!=(const LayoutInfo& rhs) const noexcept {
    return !operator==(rhs);
  }

  /// @brief Returns the minimum number of elements that are needed to fit a matrix with the given layout.
  std::size_t minMemSize() const noexcept;

  /// @brief Returns the position of the first element of the @p index tile.
  /// @pre 0 < @p index.row() < nrTiles().rows()
  /// @pre 0 < @p index.col() < nrTiles().cols()
  std::size_t tileOffset(const LocalTileIndex& index) const noexcept {
    using util::size_t::mul;
    assert(index.isValid() && index.isIn(nr_tiles_));
    return mul(index.row(), tile_offset_row_) + mul(index.col(), tile_offset_col_);
  }

  /// @brief Returns the size @p index tile.
  /// @pre 0 < @p index.row() < nrTiles().rows()
  /// @pre 0 < @p index.col() < nrTiles().cols()
  TileElementSize tileSize(const LocalTileIndex& index) const noexcept {
    assert(index.isValid() && index.isIn(nr_tiles_));
    SizeType m = std::min(block_size_.rows(), size_.rows() - index.row() * block_size_.rows());
    SizeType n = std::min(block_size_.cols(), size_.cols() - index.col() * block_size_.cols());
    return {m, n};
  }

  /// @brief Returns the minimum number of elements that are needed for the @p index tile.
  /// @pre 0 < @p index.row() < nrTiles().rows()
  /// @pre 0 < @p index.col() < nrTiles().cols()
  std::size_t minTileMemSize(const LocalTileIndex& index) const noexcept {
    assert(index.isValid() && index.isIn(nr_tiles_));
    return minTileMemSize(tileSize(index));
  }

  /// @brief Returns the minimum number of elements that are needed to fit a tile with the given size.
  /// @pre tile_size.rows() <= block_size.rows()
  /// @pre tile_size.cols() <= block_size.cols()
  std::size_t minTileMemSize(const TileElementSize& tile_size) const noexcept;

  const LocalElementSize& size() const noexcept {
    return size_;
  }

  const LocalTileSize& nrTiles() const noexcept {
    return nr_tiles_;
  }

  const TileElementSize& blockSize() const noexcept {
    return block_size_;
  }

  SizeType ldTile() const noexcept {
    return ld_tile_;
  }

private:
  LocalElementSize size_;
  LocalTileSize nr_tiles_;
  TileElementSize block_size_;

  SizeType ld_tile_;
  std::size_t tile_offset_row_;
  std::size_t tile_offset_col_;
};

/// Returns LayoutInfo for a local column major matrix.
inline LayoutInfo colMajorLayout(const LocalElementSize& size, const TileElementSize& block_size,
                                 SizeType ld) {
  using util::size_t::mul;
  return LayoutInfo(size, block_size, ld, static_cast<std::size_t>(block_size.rows()),
                    mul(block_size.cols(), ld));
}
/// Returns LayoutInfo for a distributed column major matrix.
inline LayoutInfo colMajorLayout(const matrix::Distribution& distribution, SizeType ld) {
  return colMajorLayout(distribution.localSize(), distribution.blockSize(), ld);
}

/// Returns LayoutInfo for a local matrix which use the tile layout (Advanced interface).
inline LayoutInfo tileLayout(const LocalElementSize& size, const TileElementSize& block_size,
                             SizeType ld_tile, SizeType tiles_per_col) {
  using util::size_t::mul;
  std::size_t tile_size = mul(ld_tile, block_size.cols());
  std::size_t row_offset = std::max<std::size_t>(1, tile_size);
  std::size_t col_offset = std::max<std::size_t>(1, mul(tile_size, tiles_per_col));
  return LayoutInfo(size, block_size, ld_tile, row_offset, col_offset);
}

/// Returns LayoutInfo for a distributed matrix which use the tile layout (Advanced interface).
inline LayoutInfo tileLayout(const matrix::Distribution& distribution, SizeType ld_tile,
                             SizeType tiles_per_col) {
  return tileLayout(distribution.localSize(), distribution.blockSize(), ld_tile, tiles_per_col);
}

/// Returns LayoutInfo for a local matrix which use the tile layout (Basic interface).
inline LayoutInfo tileLayout(const LocalElementSize& size, const TileElementSize& block_size) {
  SizeType ld = std::max(1, block_size.rows());
  SizeType tiles_per_col = util::ceilDiv(size.rows(), block_size.rows());
  return tileLayout(size, block_size, ld, tiles_per_col);
}

/// Returns LayoutInfo for a distributed matrix which use the tile layout (Basic interface).
inline LayoutInfo tileLayout(const matrix::Distribution& distribution) {
  return tileLayout(distribution.localSize(), distribution.blockSize());
}
}
}
