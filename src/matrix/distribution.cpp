//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix/distribution.h"

namespace dlaf {
namespace matrix {
Distribution::Distribution() noexcept
    : size_(0, 0), local_size_(0, 0), global_nr_tiles_(0, 0), local_nr_tiles_(0, 0), block_size_(1, 1),
      tile_size_(1, 1), rank_index_(0, 0), grid_size_(1, 1), source_rank_index_(0, 0) {}

Distribution::Distribution(const LocalElementSize& size, const TileElementSize& block_size)
    : size_(0, 0), local_size_(size), global_nr_tiles_(0, 0), local_nr_tiles_(0, 0),
      block_size_(block_size), tile_size_(block_size), rank_index_(0, 0), grid_size_(1, 1),
      source_rank_index_(0, 0) {
  DLAF_ASSERT(local_size_.isValid(), local_size_);
  DLAF_ASSERT(!block_size_.isEmpty(), block_size_);

  computeLocalNrTiles();
  computeGlobalSizeForNonDistr();
  computeGlobalNrTiles();
}

Distribution::Distribution(const GlobalElementSize& size, const TileElementSize& block_size,
                           const comm::Size2D& grid_size, const comm::Index2D& rank_index,
                           const comm::Index2D& source_rank_index)
    : Distribution(size, block_size, block_size, grid_size, rank_index, source_rank_index) {}

Distribution::Distribution(const GlobalElementSize& size, const TileElementSize& block_size,
                           const TileElementSize& tile_size, const comm::Size2D& grid_size,
                           const comm::Index2D& rank_index, const comm::Index2D& source_rank_index)
    : size_(size), local_size_(0, 0), global_nr_tiles_(0, 0), local_nr_tiles_(0, 0),
      block_size_(block_size), tile_size_(tile_size), rank_index_(rank_index), grid_size_(grid_size),
      source_rank_index_(source_rank_index) {
  DLAF_ASSERT(size_.isValid(), size_);
  DLAF_ASSERT(!block_size_.isEmpty(), block_size_);
  DLAF_ASSERT(!tile_size_.isEmpty(), tile_size_);
  DLAF_ASSERT(block_size_.rows() % tile_size_.rows() == 0 && block_size_.cols() % tile_size_.cols() == 0,
              block_size_, tile_size_);
  DLAF_ASSERT(!grid_size_.isEmpty(), grid_size_);
  DLAF_ASSERT(rank_index.isIn(grid_size_), rank_index, grid_size_);
  DLAF_ASSERT(source_rank_index.isIn(grid_size_), source_rank_index, grid_size_);

  computeGlobalAndLocalNrTilesAndLocalSize();
}

Distribution::Distribution(Distribution&& rhs) noexcept : Distribution(rhs) {
  // use the copy constructor and set default sizes.
  rhs.setDefaultSizes();
}

Distribution& Distribution::operator=(Distribution&& rhs) noexcept {
  // use the copy assignment and set default sizes.
  *this = rhs;

  rhs.setDefaultSizes();
  return *this;
}

void Distribution::computeGlobalSizeForNonDistr() noexcept {
  size_ = GlobalElementSize(local_size_.rows(), local_size_.cols());
}

void Distribution::computeGlobalNrTiles() noexcept {
  global_nr_tiles_ = {util::ceilDiv(size_.rows(), tile_size_.rows()),
                      util::ceilDiv(size_.cols(), tile_size_.cols())};
}

void Distribution::computeGlobalAndLocalNrTilesAndLocalSize() noexcept {
  using util::matrix::nextLocalTileFromGlobalTile;
  using util::matrix::rankGlobalTile;

  // Set global_nr_tiles_.
  computeGlobalNrTiles();

  const auto tile_row =
      nextLocalTileFromGlobalTile(global_nr_tiles_.rows(), tilesPerBlock<Coord::Row>(),
                                  grid_size_.rows(), rank_index_.row(), source_rank_index_.row());
  const auto tile_col =
      nextLocalTileFromGlobalTile(global_nr_tiles_.cols(), tilesPerBlock<Coord::Col>(),
                                  grid_size_.cols(), rank_index_.col(), source_rank_index_.col());

  // Set local_nr_tiles_.
  local_nr_tiles_ = {tile_row, tile_col};

  // The local size is computed in the following way:
  // If the last element belongs to my rank:
  //   local_size = (local_nr_tiles - 1) * tile_size + size of last tile.
  // otherwise:
  //   local_size = local_nr_tiles * tile_size
  SizeType row = 0;
  if (size_.rows() > 0) {
    if (rank_index_.row() == rankGlobalTile(global_nr_tiles_.rows() - 1, tilesPerBlock<Coord::Row>(),
                                            grid_size_.rows(), source_rank_index_.row())) {
      const auto last_tile_rows = (size_.rows() - 1) % tile_size_.rows() + 1;
      row = (tile_row - 1) * tile_size_.rows() + last_tile_rows;
    }
    else {
      row = tile_row * tile_size_.rows();
    }
  }
  SizeType col = 0;
  if (size_.cols() > 0) {
    if (rank_index_.col() == rankGlobalTile(global_nr_tiles_.cols() - 1, tilesPerBlock<Coord::Col>(),
                                            grid_size_.cols(), source_rank_index_.col())) {
      const auto last_tile_cols = (size_.cols() - 1) % tile_size_.cols() + 1;
      col = (tile_col - 1) * tile_size_.cols() + last_tile_cols;
    }
    else {
      col = tile_col * tile_size_.cols();
    }
  }

  // Set local_size_.
  local_size_ = LocalElementSize(row, col);
}

void Distribution::computeLocalNrTiles() noexcept {
  local_nr_tiles_ = {util::ceilDiv(local_size_.rows(), tile_size_.rows()),
                     util::ceilDiv(local_size_.cols(), tile_size_.cols())};
}

void Distribution::setDefaultSizes() noexcept {
  size_ = {0, 0};
  local_size_ = {0, 0};
  global_nr_tiles_ = {0, 0};
  local_nr_tiles_ = {0, 0};
  block_size_ = {1, 1};
  tile_size_ = {1, 1};

  rank_index_ = {0, 0};
  grid_size_ = {1, 1};
  source_rank_index_ = {0, 0};
}

}
}
