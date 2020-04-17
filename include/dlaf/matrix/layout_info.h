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
  /// @throw std::invalid_argument if @p tile_ld < 1 or @p tile_offset_row < 1 or @p tile_offset_col < 1,
  /// @throw std::invalid_argument if @p !size.isEmpty() and
  ///        @p tile_ld < @c max(@c size.rows(), @c block_size.rows())
  /// @throw std::invalid_argument if @p !size.isEmpty() and @p tile_row_offset < @c block_size.rows()
  /// @throw std::invalid_argument if @p !size.isEmpty() and @p tile_col_offset < size of the memory
  ///        (in elements, padding included) to store a column of tiles,
  /// @throw std::invalid_argument if the tiles overlap (combinations of @p tile_ld, @p tile_row_offset).
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

  /// Returns the minimum number of elements that are needed to fit a matrix with the given layout.
  std::size_t minMemSize() const noexcept;

  /// Returns the position of the first element of the @p index tile.
  ///
  /// @pre 0 < @p index.row() < nrTiles().rows()
  /// @pre 0 < @p index.col() < nrTiles().cols()
  std::size_t tileOffset(const LocalTileIndex& index) const noexcept {
    using util::size_t::mul;
    assert(index.isValid() && index.isIn(nr_tiles_));
    return mul(index.row(), tile_offset_row_) + mul(index.col(), tile_offset_col_);
  }

  /// Returns the size @p index tile.
  ///
  /// @pre 0 < @p index.row() < nrTiles().rows()
  /// @pre 0 < @p index.col() < nrTiles().cols()
  TileElementSize tileSize(const LocalTileIndex& index) const noexcept {
    assert(index.isValid() && index.isIn(nr_tiles_));
    SizeType m = std::min(block_size_.rows(), size_.rows() - index.row() * block_size_.rows());
    SizeType n = std::min(block_size_.cols(), size_.cols() - index.col() * block_size_.cols());
    return {m, n};
  }

  /// Returns the minimum number of elements that are needed for the @p index tile.
  ///
  /// @pre 0 < @p index.row() < nrTiles().rows()
  /// @pre 0 < @p index.col() < nrTiles().cols()
  std::size_t minTileMemSize(const LocalTileIndex& index) const noexcept {
    assert(index.isValid() && index.isIn(nr_tiles_));
    return minTileMemSize(tileSize(index));
  }

  /// Returns the minimum number of elements that are needed to fit a tile with the given size.
  ///
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
  SizeType ld = std::max<SizeType>(1, block_size.rows());
  SizeType tiles_per_col = util::ceilDiv(size.rows(), block_size.rows());
  return tileLayout(size, block_size, ld, tiles_per_col);
}

/// Returns LayoutInfo for a distributed matrix which use the tile layout (Basic interface).
inline LayoutInfo tileLayout(const matrix::Distribution& distribution) {
  return tileLayout(distribution.localSize(), distribution.blockSize());
}
}
}
