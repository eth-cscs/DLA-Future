//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cstdlib>
#include <utility>

#include <dlaf/matrix/distribution.h>

namespace dlaf::matrix {
Distribution::Distribution() noexcept
    : offset_(0, 0), size_(0, 0), local_size_(0, 0), nr_tiles_(0, 0), local_nr_tiles_(0, 0),
      block_size_(1, 1), tile_size_(1, 1), rank_index_(0, 0), grid_size_(1, 1),
      source_rank_index_(0, 0) {}

Distribution::Distribution(const LocalElementSize& size, const TileElementSize& tile_size,
                           const GlobalElementIndex& element_offset)
    : offset_(element_offset.row(), element_offset.col()), size_(0, 0), local_size_(size),
      nr_tiles_(0, 0), local_nr_tiles_(0, 0), block_size_({tile_size.rows(), tile_size.cols()}),
      tile_size_(tile_size), rank_index_(0, 0), grid_size_(1, 1), source_rank_index_(0, 0) {
  DLAF_ASSERT(local_size_.isValid(), local_size_);
  DLAF_ASSERT(!tile_size_.isEmpty(), tile_size_);

  normalize_source_rank_and_offset();
  size_ = GlobalElementSize{local_size_.rows(), local_size_.cols()};
  compute_global_nr_tiles();
  local_nr_tiles_ = LocalTileSize{nr_tiles_.rows(), nr_tiles_.cols()};
}

Distribution::Distribution(const GlobalElementSize& size, const TileElementSize& tile_size,
                           const comm::Size2D& grid_size, const comm::Index2D& rank_index,
                           const comm::Index2D& source_rank_index,
                           const GlobalElementIndex& element_offset)
    : Distribution(size, {tile_size.rows(), tile_size.cols()}, tile_size, grid_size, rank_index,
                   source_rank_index, element_offset) {}

Distribution::Distribution(const GlobalElementSize& size, const TileElementSize& tile_size,
                           const comm::Size2D& grid_size, const comm::Index2D& rank_index,
                           const comm::Index2D& source_rank_index, const GlobalTileIndex& tile_offset,
                           const GlobalElementIndex& element_offset)
    : Distribution(size, tile_size, grid_size, rank_index, source_rank_index,
                   GlobalElementIndex(tile_offset.row() * tile_size.rows() + element_offset.row(),
                                      tile_offset.col() * tile_size.cols() + element_offset.col())) {}

Distribution::Distribution(const GlobalElementSize& size, const GlobalElementSize& block_size,
                           const TileElementSize& tile_size, const comm::Size2D& grid_size,
                           const comm::Index2D& rank_index, const comm::Index2D& source_rank_index,
                           const GlobalElementIndex& element_offset)
    : offset_(element_offset), size_(size), local_size_(0, 0), nr_tiles_(0, 0), local_nr_tiles_(0, 0),
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

  normalize_source_rank_and_offset();
  compute_global_nr_tiles();
  compute_local_nr_tiles_and_local_size();
}

Distribution::Distribution(const GlobalElementSize& size, const GlobalElementSize& block_size,
                           const TileElementSize& tile_size, const comm::Size2D& grid_size,
                           const comm::Index2D& rank_index, const comm::Index2D& source_rank_index,
                           const GlobalTileIndex& tile_offset, const GlobalElementIndex& element_offset)
    : Distribution(size, block_size, tile_size, grid_size, rank_index, source_rank_index,
                   GlobalElementIndex(tile_offset.row() * tile_size.rows() + element_offset.row(),
                                      tile_offset.col() * tile_size.cols() + element_offset.col())) {}

Distribution::Distribution(Distribution&& rhs) noexcept : Distribution(rhs) {
  // use the copy constructor and set default sizes.
  rhs.set_default_sizes();
}

Distribution& Distribution::operator=(Distribution&& rhs) noexcept {
  // use the copy assignment and set default sizes.
  *this = rhs;

  rhs.set_default_sizes();
  return *this;
}

Distribution::Distribution(Distribution rhs, const SubDistributionSpec& spec)
    : Distribution(std::move(rhs)) {
  DLAF_ASSERT(spec.origin.isValid(), spec.origin);
  DLAF_ASSERT(spec.size.isValid(), spec.size);
  DLAF_ASSERT(spec.origin.row() + spec.size.rows() <= size_.rows(), spec.origin, spec.size, size_);
  DLAF_ASSERT(spec.origin.col() + spec.size.cols() <= size_.cols(), spec.origin, spec.size, size_);

  offset_ = offset_ + sizeFromOrigin(spec.origin);
  size_ = spec.size;

  normalize_source_rank_and_offset();
  compute_global_nr_tiles();
  compute_local_nr_tiles_and_local_size();
}

void Distribution::compute_global_nr_tiles() noexcept {
  nr_tiles_ = {size_.rows() > 0 ? util::ceilDiv(size_.rows() + global_tile_element_offset<Coord::Row>(),
                                                tile_size_.rows())
                                : 0,
               size_.cols() > 0 ? util::ceilDiv(size_.cols() + global_tile_element_offset<Coord::Col>(),
                                                tile_size_.cols())
                                : 0};
}

void Distribution::compute_local_nr_tiles_and_local_size() noexcept {
  const auto tile_row = next_local_tile_from_global_tile<Coord::Row>(nr_tiles_.rows());
  const auto tile_col = next_local_tile_from_global_tile<Coord::Col>(nr_tiles_.cols());

  // Set local_nr_tiles_.
  local_nr_tiles_ = {tile_row, tile_col};

  SizeType row = compute_local_size<Coord::Row>();
  SizeType col = compute_local_size<Coord::Col>();

  // Set local_size_.
  local_size_ = {row, col};
}

template <Coord rc>
SizeType Distribution::compute_local_size() noexcept {
  if (local_nr_tiles_.get<rc>() == 0)
    return 0;

  // Start from full tiles
  SizeType ret = local_nr_tiles_.get<rc>() * tile_size_.get<rc>();

  // Fix first tile size removing local offset
  ret -= local_tile_element_offset<rc>();

  // Fix last tile size
  if (rank_index_.get<rc>() == rank_global_tile<rc>(nr_tiles_.get<rc>() - 1))
    // remove the elements missing in the last tile
    ret -= nr_tiles_.get<rc>() * tile_size_.get<rc>() -
           (size_.get<rc>() + global_tile_element_offset<rc>());

  return ret;
}

void Distribution::normalize_source_rank_and_offset() noexcept {
  auto div_row = std::div(offset_.row(), block_size_.rows());
  auto div_col = std::div(offset_.col(), block_size_.cols());

  offset_ = {div_row.rem, div_col.rem};
  source_rank_index_ = {(source_rank_index_.row() + static_cast<int>(div_row.quot)) % grid_size_.rows(),
                        (source_rank_index_.col() + static_cast<int>(div_col.quot)) % grid_size_.cols()};

  DLAF_ASSERT(offset_.row() < block_size_.rows(), offset_, block_size_);
  DLAF_ASSERT(offset_.col() < block_size_.cols(), offset_, block_size_);
}

void Distribution::set_default_sizes() noexcept {
  offset_ = {0, 0};
  size_ = {0, 0};
  local_size_ = {0, 0};
  nr_tiles_ = {0, 0};
  local_nr_tiles_ = {0, 0};
  block_size_ = {1, 1};
  tile_size_ = {1, 1};

  rank_index_ = {0, 0};
  grid_size_ = {1, 1};
  source_rank_index_ = {0, 0};
}

namespace internal {
Distribution get_single_tile_per_block_distribution(const Distribution& dist) {
  Distribution ret = dist;
  ret.tile_size_ = {dist.block_size_.rows(), dist.block_size_.cols()};
  auto nr_blocks = dist.nr_blocks();
  ret.nr_tiles_ = {nr_blocks.rows(), nr_blocks.cols()};
  auto local_nr_blocks = dist.local_nr_blocks();
  ret.local_nr_tiles_ = {local_nr_blocks.rows(), local_nr_blocks.cols()};
  return ret;
}
}
}
