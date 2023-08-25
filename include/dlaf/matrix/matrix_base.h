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

/// @file

#include <cstddef>
#include <ostream>

#include <dlaf/matrix/distribution.h>

namespace dlaf {
namespace matrix {
namespace internal {

class MatrixBase {
public:
  MatrixBase(Distribution distribution) : distribution_(std::move(distribution)) {}

  MatrixBase(const Distribution& distribution, const LocalTileSize& tiles_per_block)
      : distribution_(distribution.size(), distribution.blockSize(),
                      TileElementSize{distribution.blockSize().rows() / tiles_per_block.rows(),
                                      distribution.blockSize().cols() / tiles_per_block.cols()},
                      distribution.commGridSize(), distribution.rankIndex(),
                      distribution.sourceRankIndex()) {
    DLAF_ASSERT(distribution.blockSize() == distribution.baseTileSize(),
                "distribution should be the distribution of the original Matrix.",
                distribution.blockSize(), distribution.baseTileSize());
    DLAF_ASSERT(distribution.blockSize() == distribution_.blockSize(), distribution.blockSize(),
                distribution_.blockSize());
  }

  MatrixBase(const MatrixBase& rhs) = default;
  MatrixBase& operator=(const MatrixBase& rhs) = default;

  /// Returns the global size in elements of the matrix.
  const GlobalElementSize& size() const noexcept {
    return distribution_.size();
  }

  /// Returns the block size of the matrix.
  const TileElementSize& blockSize() const noexcept {
    return distribution_.blockSize();
  }

  /// Returns the complete tile size of the matrix.
  const TileElementSize& baseTileSize() const noexcept {
    return distribution_.baseTileSize();
  }

  /// Returns the number of tiles of the global matrix (2D size).
  const GlobalTileSize& nrTiles() const noexcept {
    return distribution_.nrTiles();
  }

  /// Returns the id associated to the matrix of this rank.
  const comm::Index2D& rankIndex() const noexcept {
    return distribution_.rankIndex();
  }

  ///
  const comm::Index2D& sourceRankIndex() const noexcept {
    return distribution_.sourceRankIndex();
  }

  /// Returns the size of the communicator grid associated to the matrix.
  const comm::Size2D& commGridSize() const noexcept {
    return distribution_.commGridSize();
  }

  /// Returns the 2D rank index of the process that stores the tile with global index @p global_tile.
  ///
  /// @pre global_tile.isIn(nrTiles()).
  comm::Index2D rankGlobalTile(const GlobalTileIndex& global_tile) const noexcept {
    return distribution_.rankGlobalTile(global_tile);
  }

  /// Returns the distribution of the matrix.
  const matrix::Distribution& distribution() const noexcept {
    return distribution_;
  }

  /// Returns the size of the Tile with global index @p index.
  TileElementSize tileSize(const GlobalTileIndex& index) const noexcept {
    return distribution_.tileSize(index);
  }

protected:
  // move constructor and assignment are protected to avoid invalidation of objects of
  // derived classes. I.e.:
  // class MatrixDerived : public MatrixBase;
  // MatrixDerived derived;
  // MatrixBase base = std::move(derived);
  MatrixBase(MatrixBase&& rhs) = default;
  MatrixBase& operator=(MatrixBase&& rhs) = default;

  /// Returns the position in the vector of the index Tile.
  ///
  /// @pre index.isIn(localNrTiles()).
  std::size_t tileLinearIndex(const LocalTileIndex& index) const noexcept {
    DLAF_ASSERT_MODERATE(index.isIn(distribution_.localNrTiles()), index, distribution_.localNrTiles());
    return to_sizet(index.row() + distribution_.localNrTiles().rows() * index.col());
  }

  /// Prints information about the matrix.
  friend std::ostream& operator<<(std::ostream& out, const MatrixBase& matrix) {
    // clang-format off
    return out << "size="         << matrix.size()
               << ", block_size=" << matrix.blockSize()
               << ", tiles_grid=" << matrix.nrTiles()
               << ", rank_index=" << matrix.rankIndex()
               << ", comm_grid="  << matrix.commGridSize();
    // clang-format on
  }

private:
  Distribution distribution_;
};

}
}
}
