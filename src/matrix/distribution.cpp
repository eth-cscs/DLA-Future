//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
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
      rank_index_(0, 0), grid_size_(1, 1), source_rank_index_(0, 0) {}

Distribution::Distribution(const LocalElementSize& size, const TileElementSize& block_size)
    : size_(0, 0), local_size_(size), global_nr_tiles_(0, 0), local_nr_tiles_(0, 0),
      block_size_(block_size), rank_index_(0, 0), grid_size_(1, 1), source_rank_index_(0, 0) {
  DLAF_ASSERT((local_size_.isValid()), "Invalid Matrix size");
  DLAF_ASSERT((block_size_.isValid() && !block_size_.isEmpty()), "Invalid Block size");

  computeLocalNrTiles(local_size_, block_size_);
  computeGlobalSizeForNonDistr(local_size_);
  computeGlobalNrTiles(size_, block_size_);
}

Distribution::Distribution(const GlobalElementSize& size, const TileElementSize& block_size,
                           const comm::Size2D& grid_size, const comm::Index2D& rank_index,
                           const comm::Index2D& source_rank_index)
    : size_(size), local_size_(0, 0), global_nr_tiles_(0, 0), local_nr_tiles_(0, 0),
      block_size_(block_size), rank_index_(rank_index), grid_size_(grid_size),
      source_rank_index_(source_rank_index) {
  DLAF_ASSERT((size_.isValid()), "Invalid Matrix size");
  DLAF_ASSERT((block_size_.isValid() && !block_size_.isEmpty()), "Invalid Block size");
  DLAF_ASSERT((grid_size_.isValid() && !grid_size_.isEmpty()), "Invalid Communicator Size");
  DLAF_ASSERT((rank_index_.isValid() && rank_index.isIn(grid_size_)), "Invalid Rank Index");
  DLAF_ASSERT((source_rank_index_.isValid() && source_rank_index.isIn(grid_size_)),
              "Invalid Matrix Source Rank Index");

  computeGlobalAndLocalNrTilesAndLocalSize(size_, block_size_, grid_size_, rank_index_,
                                           source_rank_index_);
}

Distribution::Distribution(Distribution&& rhs) noexcept
    : size_(rhs.size_), local_size_(rhs.local_size_), global_nr_tiles_(rhs.global_nr_tiles_),
      local_nr_tiles_(rhs.local_nr_tiles_), block_size_(rhs.block_size_), rank_index_(rhs.rank_index_),
      grid_size_(rhs.grid_size_), source_rank_index_(rhs.source_rank_index_) {
  rhs.setDefaultSizes();
}

Distribution& Distribution::operator=(Distribution&& rhs) noexcept {
  size_ = rhs.size_;
  local_size_ = rhs.local_size_;
  global_nr_tiles_ = rhs.global_nr_tiles_;
  local_nr_tiles_ = rhs.local_nr_tiles_;
  block_size_ = rhs.block_size_;
  rank_index_ = rhs.rank_index_;
  grid_size_ = rhs.grid_size_;
  source_rank_index_ = rhs.source_rank_index_;

  rhs.setDefaultSizes();
  return *this;
}

void Distribution::computeGlobalSizeForNonDistr(const LocalElementSize& local_size) noexcept {
  size_ = GlobalElementSize(local_size.rows(), local_size.cols());
}

void Distribution::computeGlobalNrTiles(const GlobalElementSize& size,
                                        const TileElementSize& block_size) noexcept {
  global_nr_tiles_ = {util::ceilDiv(size.rows(), block_size.rows()),
                      util::ceilDiv(size.cols(), block_size.cols())};
}

void Distribution::computeGlobalAndLocalNrTilesAndLocalSize(
    const GlobalElementSize& size, const TileElementSize& block_size, const comm::Size2D& grid_size,
    const comm::Index2D& rank_index, const comm::Index2D& source_rank_index) noexcept {
  // Set global_nr_tiles_.
  computeGlobalNrTiles(size, block_size);

  auto tile_row = util::matrix::nextLocalTileFromGlobalTile(global_nr_tiles_.rows(), grid_size.rows(),
                                                            rank_index.row(), source_rank_index.row());
  auto tile_col = util::matrix::nextLocalTileFromGlobalTile(global_nr_tiles_.cols(), grid_size.cols(),
                                                            rank_index.col(), source_rank_index.col());

  // Set local_nr_tiles_.
  local_nr_tiles_ = {tile_row, tile_col};

  // The local size is computed in the following way:
  // If the last element belongs to my rank:
  //   local_size = (local_nr_tiles - 1) * block_size + size of last tile.
  // otherwise:
  //   local_size = local_nr_tiles * block_size
  SizeType row = 0;
  if (size.rows() > 0) {
    if (rank_index.row() == util::matrix::rankGlobalTile(global_nr_tiles_.rows() - 1, grid_size.rows(),
                                                         source_rank_index.row())) {
      auto last_tile_rows = (size.rows() - 1) % block_size.rows() + 1;
      row = (tile_row - 1) * block_size.rows() + last_tile_rows;
    }
    else {
      row = tile_row * block_size.rows();
    }
  }
  SizeType col = 0;
  if (size.cols() > 0) {
    if (rank_index.col() == util::matrix::rankGlobalTile(global_nr_tiles_.cols() - 1, grid_size.cols(),
                                                         source_rank_index.col())) {
      auto last_tile_cols = (size.cols() - 1) % block_size.cols() + 1;
      col = (tile_col - 1) * block_size.cols() + last_tile_cols;
    }
    else {
      col = tile_col * block_size.cols();
    }
  }

  // Set local_size_.
  local_size_ = LocalElementSize(row, col);
}

void Distribution::computeLocalNrTiles(const LocalElementSize& local_size,
                                       const TileElementSize& block_size) noexcept {
  local_nr_tiles_ = {util::ceilDiv(local_size.rows(), block_size.rows()),
                     util::ceilDiv(local_size.cols(), block_size.cols())};
}

void Distribution::setDefaultSizes() noexcept {
  size_ = {0, 0};
  local_size_ = {0, 0};
  global_nr_tiles_ = {0, 0};
  local_nr_tiles_ = {0, 0};
  block_size_ = {1, 1};

  rank_index_ = {0, 0};
  grid_size_ = {1, 1};
  source_rank_index_ = {0, 0};
}

}
}
