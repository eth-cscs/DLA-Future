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

#ifndef DLAF_MATRIX_ENABLE_DEPRECATED
#define DLAF_MATRIX_ENABLE_DEPRECATED 0
#endif
#if (DLAF_MATRIX_ENABLE_DEPRECATED)
#define DLAF_MATRIX_DEPRECATED(x) [[deprecated(x)]]
#else
#define DLAF_MATRIX_DEPRECATED(x)
#endif

namespace dlaf {
namespace matrix {
namespace internal {

class MatrixBase {
public:
  MatrixBase(Distribution distribution) : distribution_(std::move(distribution)) {}

  MatrixBase(const Distribution& distribution, const LocalTileSize& tiles_per_block)
      : distribution_(distribution.size(), distribution.block_size(),
                      TileElementSize{distribution.block_size().rows() / tiles_per_block.rows(),
                                      distribution.block_size().cols() / tiles_per_block.cols()},
                      distribution.grid_size(), distribution.rank_index(),
                      distribution.source_rank_index()) {
    DLAF_ASSERT(distribution.block_size() == distribution.tile_size(),
                "distribution should be the distribution of the original Matrix.",
                distribution.block_size(), distribution.tile_size());
    DLAF_ASSERT(distribution.block_size() == distribution_.block_size(), distribution.block_size(),
                distribution_.block_size());
  }

  MatrixBase(const MatrixBase& rhs) = default;
  MatrixBase& operator=(const MatrixBase& rhs) = default;

  /// Returns the global size in elements of the matrix.
  const GlobalElementSize& size() const noexcept {
    return distribution_.size();
  }

  /// Returns the complete block size of the matrix.
  const TileElementSize& block_size() const noexcept {
    return distribution_.block_size();
  }

  /// Returns the complete tile size of the matrix.
  const TileElementSize& tile_size() const noexcept {
    return distribution_.tile_size();
  }

  /// Returns the number of tiles of the global matrix (2D size).
  const GlobalTileSize& nr_tiles() const noexcept {
    return distribution_.nr_tiles();
  }

  /// Returns the id associated to the matrix of this rank.
  const comm::Index2D& rank_index() const noexcept {
    return distribution_.rank_index();
  }

  /// Returns the 2D rank index of the process that stores the top-left tile of the matrix
  const comm::Index2D& source_rank_index() const noexcept {
    return distribution_.source_rank_index();
  }

  /// Returns the size of the communicator grid associated to the matrix.
  const comm::Size2D& grid_size() const noexcept {
    return distribution_.grid_size();
  }

  /// Returns the 2D rank index of the process that stores the tile with global index @p global_tile.
  ///
  /// @pre global_tile.isIn(nrTiles()).
  comm::Index2D rank_global_tile(const GlobalTileIndex& global_tile) const noexcept {
    return distribution_.rank_global_tile(global_tile);
  }

  /// Returns the size of the Tile with global index @p index.
  TileElementSize tile_size_of(const GlobalTileIndex& index) const noexcept {
    return distribution_.tile_size_of(index);
  }

  /// Returns the distribution of the matrix.
  const matrix::Distribution& distribution() const noexcept {
    return distribution_;
  }

  DLAF_MATRIX_DEPRECATED("method has been renamed in snake case")
  const TileElementSize& blockSize() const noexcept {
    return distribution_.blockSize();
  }

  DLAF_MATRIX_DEPRECATED("Use tile_size method")
  const TileElementSize& baseTileSize() const noexcept {
    return distribution_.baseTileSize();
  }

  DLAF_MATRIX_DEPRECATED("method has been renamed in snake case")
  const GlobalTileSize& nrTiles() const noexcept {
    return distribution_.nrTiles();
  }

  DLAF_MATRIX_DEPRECATED("method has been renamed in snake case")
  const comm::Index2D& rankIndex() const noexcept {
    return distribution_.rankIndex();
  }

  DLAF_MATRIX_DEPRECATED("method has been renamed in snake case")
  const comm::Index2D& sourceRankIndex() const noexcept {
    return distribution_.sourceRankIndex();
  }

  DLAF_MATRIX_DEPRECATED("method has been renamed in snake case")
  const comm::Size2D& commGridSize() const noexcept {
    return distribution_.commGridSize();
  }

  DLAF_MATRIX_DEPRECATED("method has been renamed in snake case")
  comm::Index2D rankGlobalTile(const GlobalTileIndex& global_tile) const noexcept {
    return distribution_.rankGlobalTile(global_tile);
  }

  DLAF_MATRIX_DEPRECATED("Use tile_size_of method")
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
  std::size_t tile_linear_index(const LocalTileIndex& index) const noexcept {
    DLAF_ASSERT_MODERATE(index.isIn(distribution_.local_nr_tiles()), index,
                         distribution_.local_nr_tiles());
    return to_sizet(index.row() + distribution_.local_nr_tiles().rows() * index.col());
  }

  DLAF_MATRIX_DEPRECATED("method has been renamed in snake case")
  std::size_t tileLinearIndex(const LocalTileIndex& index) const noexcept {
    return tile_linear_index(index);
  }

  /// Prints information about the matrix.
  friend std::ostream& operator<<(std::ostream& out, const MatrixBase& matrix) {
    // clang-format off
    return out << "size="         << matrix.size()
               << ", block_size=" << matrix.block_size()
               << ", tile_size="  << matrix.tile_size()
               << ", tiles_grid=" << matrix.nr_tiles()
               << ", rank_index=" << matrix.rank_index()
               << ", comm_grid="  << matrix.grid_size()
               << ", src_rank="   << matrix.distribution().source_rank_index()
               << ", offset="     << matrix.distribution().offset();
    // clang-format on
  }

private:
  Distribution distribution_;
};

}
}
}
