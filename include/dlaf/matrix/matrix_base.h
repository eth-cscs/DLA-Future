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

#include <ostream>

#include "dlaf/matrix/distribution.h"

namespace dlaf {
namespace matrix {
namespace internal {

class MatrixBase {
public:
  MatrixBase(Distribution&& distribution)
      : distribution_(std::make_shared<Distribution>(std::move(distribution))) {}

  MatrixBase(const MatrixBase& rhs) = default;
  MatrixBase& operator=(const MatrixBase& rhs) = default;

  /// Returns the global size in elements of the matrix.
  const GlobalElementSize& size() const noexcept {
    return distribution_->size();
  }

  /// Returns the block size of the matrix.
  const TileElementSize& blockSize() const noexcept {
    return distribution_->blockSize();
  }

  /// Returns the number of tiles of the global matrix (2D size).
  const GlobalTileSize& nrTiles() const noexcept {
    return distribution_->nrTiles();
  }

  /// Returns the id associated to the matrix of this rank.
  const comm::Index2D& rankIndex() const noexcept {
    return distribution_->rankIndex();
  }

  /// Returns the size of the communicator grid associated to the matrix.
  const comm::Size2D& commGridSize() const noexcept {
    return distribution_->commGridSize();
  }

  /// Returns the 2D rank index of the process that stores the tile with global index @p global_tile.
  ///
  /// @pre global_tile.isIn(nrTiles()).
  comm::Index2D rankGlobalTile(const GlobalTileIndex& global_tile) const noexcept {
    return distribution_->rankGlobalTile(global_tile);
  }

  /// Returns the distribution of the matrix.
  const matrix::Distribution& distribution() const noexcept {
    return *distribution_;
  }

  /// Returns the size of the Tile with global index @p index.
  TileElementSize tileSize(const GlobalTileIndex& index) noexcept {
    return {std::min(blockSize().rows(), size().rows() - index.row() * blockSize().rows()),
            std::min(blockSize().cols(), size().cols() - index.col() * blockSize().cols())};
  }

protected:
  // move constructor and assignment are protected to avoid invalidation of objects of
  // derived classes. I.e.:
  // class MatrixDerived : public MatrixBase;
  // MatrixDerived derived;
  // MatrixBase base = std::move(derived);
  MatrixBase(MatrixBase&& rhs) = default;
  MatrixBase& operator=(MatrixBase&& rhs) = default;

  static std::size_t futureVectorSize(const LocalTileSize& local_nr_tiles) noexcept {
    return util::size_t::mul(local_nr_tiles.rows(), local_nr_tiles.cols());
  }

  /// Returns the position in the vector of the index Tile.
  ///
  /// @pre index.isIn(localNrTiles()).
  std::size_t tileLinearIndex(const LocalTileIndex& index) const noexcept {
    DLAF_ASSERT_HEAVY(index.isIn(distribution_->localNrTiles()), "");
    using util::size_t::sum;
    using util::size_t::mul;
    return sum(index.row(), mul(distribution_->localNrTiles().rows(), index.col()));
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
  std::shared_ptr<Distribution> distribution_;
};

}
}
}
