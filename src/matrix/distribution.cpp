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
      rank_index_(0, 0), comm_size_(1, 1), source_rank_index_(0, 0) {}

Distribution::Distribution(const LocalElementSize& size, const TileElementSize& block_size)
    : size_(0, 0), local_size_(size), global_nr_tiles_(0, 0), local_nr_tiles_(0, 0),
      block_size_(block_size), rank_index_(0, 0), comm_size_(1, 1), source_rank_index_(0, 0) {
  if (!local_size_.isValid())
    throw std::invalid_argument("Error: Invalid Matrix size");
  if (!block_size_.isValid() || block_size_.isEmpty())
    throw std::invalid_argument("Error: Invalid Block size");

  computeLocalNrTiles();
  computeGlobalSize();
  computeGlobalNrTiles();
}

Distribution::Distribution(const GlobalElementSize& size, const TileElementSize& block_size,
                           const comm::Size2D& comm_size, const comm::Index2D& rank_index,
                           const comm::Index2D& source_rank_index)
    : size_(size), local_size_(0, 0), global_nr_tiles_(0, 0), local_nr_tiles_(0, 0),
      block_size_(block_size), rank_index_(rank_index), comm_size_(comm_size),
      source_rank_index_(source_rank_index) {
  if (!size_.isValid())
    throw std::invalid_argument("Error: Invalid Matrix size");
  if (!block_size_.isValid() || block_size_.isEmpty())
    throw std::invalid_argument("Error: Invalid Block size");
  if (!rank_index_.isValid())
    throw std::invalid_argument("Error: Invalid Rank Index");
  if (!comm_size_.isValid() || comm_size_.isEmpty())
    throw std::invalid_argument("Error: Invalid Communicator Size");
  if (!source_rank_index_.isValid())
    throw std::invalid_argument("Error: Invalid Matrix Source Rank Index");

  computeGlobalNrTiles();
  computeLocalSize();
  computeLocalNrTiles();
}

Distribution::Distribution(Distribution&& rhs) noexcept
    : size_(rhs.size_), local_size_(rhs.local_size_), global_nr_tiles_(rhs.global_nr_tiles_),
      local_nr_tiles_(rhs.local_nr_tiles_), block_size_(rhs.block_size_), rank_index_(rhs.rank_index_),
      comm_size_(rhs.comm_size_), source_rank_index_(rhs.source_rank_index_) {
  rhs.setDefaultSizes();
}

Distribution& Distribution::operator=(Distribution&& rhs) noexcept {
  size_ = rhs.size_;
  local_size_ = rhs.local_size_;
  global_nr_tiles_ = rhs.global_nr_tiles_;
  local_nr_tiles_ = rhs.local_nr_tiles_;
  block_size_ = rhs.block_size_;
  rank_index_ = rhs.rank_index_;
  comm_size_ = rhs.comm_size_;
  source_rank_index_ = rhs.source_rank_index_;

  rhs.setDefaultSizes();
  return *this;
}

void Distribution::computeGlobalSize() noexcept {
  assert(comm_size_ == comm::Size2D(1, 1));
  assert(rank_index_ == comm::Index2D(0, 0));
  assert(source_rank_index_ == comm::Index2D(0, 0));
  size_ = GlobalElementSize(local_size_.rows(), local_size_.cols());
}

void Distribution::computeGlobalNrTiles() noexcept {
  global_nr_tiles_ = {util::ceilDiv(size_.rows(), block_size_.rows()),
                      util::ceilDiv(size_.cols(), block_size_.cols())};
}

void Distribution::computeLocalSize() noexcept {
  auto row = nextLocalElementFromGlobalElement<RowCol::Row>(size_.rows());
  auto col = nextLocalElementFromGlobalElement<RowCol::Col>(size_.cols());
  local_size_ = LocalElementSize(row, col);
}

void Distribution::computeLocalNrTiles() noexcept {
  local_nr_tiles_ = {util::ceilDiv(local_size_.rows(), block_size_.rows()),
                     util::ceilDiv(local_size_.cols(), block_size_.cols())};
}

void Distribution::setDefaultSizes() noexcept {
  size_ = {0, 0};
  local_size_ = {0, 0};
  global_nr_tiles_ = {0, 0};
  local_nr_tiles_ = {0, 0};
  block_size_ = {1, 1};

  rank_index_ = {0, 0};
  comm_size_ = {1, 1};
  source_rank_index_ = {0, 0};
}

}
}
