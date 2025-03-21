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

#include <dlaf/common/assert.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/index.h>
#include <dlaf/util_math.h>

namespace dlaf {
namespace matrix {

/// LayoutInfo contains the information about how the elements of a matrix are stored in memory.
/// More details available in misc/matrix_distribution.md.

class LayoutInfo {
public:
  /// Construct a generic layout of a local matrix of size @p size.
  ///
  /// @param[in] size the size of the local matrix,
  /// @param[in] block_size the size of the tiles,
  /// @param[in] tile_ld the leading dimension of the tile,
  /// @param[in] tile_offset_row the distance of the next row tile,
  /// @param[in] tile_offset_col the distance of the next column tile.
  ///
  /// See misc/matrix_distribution.md for for more detail about the parameters.
  ///
  /// - If size.isEmpty():
  ///   @pre tile_ld >= 1,
  ///   @pre tile_offset_row >= 1,
  ///   @pre tile_offset_col >= 1;
  /// - if !size.isEmpty():
  ///   @pre tile_ld >= max(size.rows(), block_size.rows()),
  ///   @pre tile_offset_row >= block_size.rows(),
  ///   @pre tile_offset_col >= size of the memory (in elements, padding included)
  ///      to store a column of tiles;
  ///   @pre the tiles should not overlap (combinations of @p tile_ld, @p tile_row_offset).
  LayoutInfo(const LocalElementSize& size, const TileElementSize& block_size, SizeType tile_ld,
             SizeType tile_offset_row, SizeType tile_offset_col);

  bool operator==(const LayoutInfo& rhs) const noexcept {
    return size_ == rhs.size_ && nr_tiles_ == rhs.nr_tiles_ && block_size_ == rhs.block_size_ &&
           ld_tile_ == rhs.ld_tile_ && tile_offset_row_ == rhs.tile_offset_row_ &&
           tile_offset_col_ == rhs.tile_offset_col_;
  }

  bool operator!=(const LayoutInfo& rhs) const noexcept {
    return !operator==(rhs);
  }

  /// Returns the minimum number of elements that are needed to fit a matrix with the given layout.
  SizeType minMemSize() const noexcept;

  /// Returns the position of the first element of the @p index tile.
  ///
  /// @pre index.isIn(nr_tiles_).
  SizeType tileOffset(const LocalTileIndex& index) const noexcept {
    DLAF_ASSERT_HEAVY(index.isIn(nr_tiles_), index, nr_tiles_);
    return index.row() * tile_offset_row_ + index.col() * tile_offset_col_;
  }

  /// Returns the size @p index tile.
  ///
  /// @pre index.isIn(nr_tiles_).
  TileElementSize tileSize(const LocalTileIndex& index) const noexcept {
    DLAF_ASSERT_HEAVY(index.isIn(nr_tiles_), index, nr_tiles_);
    SizeType m = std::min(block_size_.rows(), size_.rows() - index.row() * block_size_.rows());
    SizeType n = std::min(block_size_.cols(), size_.cols() - index.col() * block_size_.cols());
    return {m, n};
  }

  /// Returns the minimum number of elements that are needed for the @p index tile.
  ///
  /// @pre 0 < @p index.row() < nrTiles().rows(),
  /// @pre 0 < @p index.col() < nrTiles().cols(),
  /// @pre index.isIn(nr_tiles_).
  SizeType minTileMemSize(const LocalTileIndex& index) const noexcept {
    DLAF_ASSERT_HEAVY(index.isIn(nr_tiles_), index, nr_tiles_);
    return minTileMemSize(tileSize(index));
  }

  /// Returns the minimum number of elements that are needed to fit a tile with the given size.
  ///
  /// @pre tile_size.rows() <= block_size.rows(),
  /// @pre tile_size.cols() <= block_size.cols().
  SizeType minTileMemSize(const TileElementSize& tile_size) const noexcept;

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
  SizeType tile_offset_row_;
  SizeType tile_offset_col_;
};

/// Returns LayoutInfo for a local column major matrix.
inline LayoutInfo colMajorLayout(const LocalElementSize& size, const TileElementSize& block_size,
                                 SizeType ld) {
  return LayoutInfo(size, block_size, ld, block_size.rows(), block_size.cols() * ld);
}
/// Returns LayoutInfo for a distributed column major matrix.
inline LayoutInfo colMajorLayout(const matrix::Distribution& distribution, SizeType ld) {
  return colMajorLayout(distribution.local_size(), distribution.block_size(), ld);
}

/// Returns LayoutInfo for a local matrix which use the tile layout (Advanced interface).
inline LayoutInfo tileLayout(const LocalElementSize& size, const TileElementSize& block_size,
                             SizeType ld_tile, SizeType tiles_per_col) {
  SizeType tile_size = static_cast<SizeType>(ld_tile) * block_size.cols();
  SizeType row_offset = std::max<SizeType>(1, tile_size);
  SizeType col_offset = std::max<SizeType>(1, tile_size * tiles_per_col);
  return LayoutInfo(size, block_size, ld_tile, row_offset, col_offset);
}

/// Returns LayoutInfo for a distributed matrix which use the tile layout (Advanced interface).
inline LayoutInfo tileLayout(const matrix::Distribution& distribution, SizeType ld_tile,
                             SizeType tiles_per_col) {
  return tileLayout(distribution.local_size(), distribution.block_size(), ld_tile, tiles_per_col);
}

/// Returns LayoutInfo for a local matrix which use the tile layout (Basic interface).
inline LayoutInfo tileLayout(const LocalElementSize& size, const TileElementSize& block_size) {
  SizeType ld = std::max<SizeType>(1, block_size.rows());
  SizeType tiles_per_col = util::ceilDiv(size.rows(), block_size.rows());
  return tileLayout(size, block_size, ld, tiles_per_col);
}

/// Returns LayoutInfo for a distributed matrix which use the tile layout (Basic interface).
inline LayoutInfo tileLayout(const matrix::Distribution& distribution) {
  return tileLayout(distribution.local_size(), distribution.block_size());
}
}
}
