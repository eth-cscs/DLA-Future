//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix/layout_info.h"

#include "dlaf/common/assert.h"
#include "dlaf/util_math.h"

namespace dlaf {
namespace matrix {
LayoutInfo::LayoutInfo(const LocalElementSize& size, const TileElementSize& block_size, SizeType tile_ld,
                       std::size_t tile_offset_row, std::size_t tile_offset_col)
    : size_(size), nr_tiles_(0, 0), block_size_(block_size), ld_tile_(tile_ld),
      tile_offset_row_(tile_offset_row), tile_offset_col_(tile_offset_col) {
  using util::size_t::sum;
  using util::size_t::mul;

  if (!size_.isValid()) {
    throw std::invalid_argument("Error: Invalid Matrix size");
  }
  if (!block_size_.isValid() || block_size_.isEmpty()) {
    throw std::invalid_argument("Error: Invalid Block size");
  }

  nr_tiles_ = {util::ceilDiv(size_.rows(), block_size_.rows()),
               util::ceilDiv(size_.cols(), block_size_.cols())};

  if (size_.isEmpty()) {
    if (ld_tile_ < 1) {
      throw std::invalid_argument("Error: Invalid Leading Dimension");
    }
    if (tile_offset_row_ < 1) {
      throw std::invalid_argument("Error: Invalid Tile Row Offset");
    }
    if (tile_offset_col_ < 1) {
      throw std::invalid_argument("Error: Invalid Tile Col Offset");
    }
  }
  else {
    SizeType last_rows = size_.rows() - block_size_.rows() * (nr_tiles_.rows() - 1);

    SizeType max_rows_tiles = std::min(size_.rows(), block_size_.rows());
    if (ld_tile_ < max_rows_tiles) {
      throw std::invalid_argument("Error: Invalid Leading Dimension");
    }
    if (tile_offset_row_ < static_cast<std::size_t>(max_rows_tiles)) {
      throw std::invalid_argument("Error: Invalid Tile Row Offset");
    }
    if (tile_offset_row_ < minTileMemSize(block_size_) &&
        static_cast<std::size_t>(ld_tile_) < sum(tileOffset({nr_tiles_.rows() - 1, 0}), last_rows)) {
      throw std::invalid_argument("Error: Invalid Leading Dimension & Tile Row Offset combination");
    }
    if (tile_offset_col_ < tileOffset({nr_tiles_.rows() - 1, 0}) +
                               minTileMemSize(LocalTileIndex(nr_tiles_.rows() - 1, 0))) {
      throw std::invalid_argument("Error: Invalid Tile Col Offset");
    }
  }
}

std::size_t LayoutInfo::minMemSize() const noexcept {
  if (size_.isEmpty()) {
    return 0;
  }

  LocalTileIndex index_last(nr_tiles_.rows() - 1, nr_tiles_.cols() - 1);
  return tileOffset(index_last) + minTileMemSize(index_last);
}

std::size_t LayoutInfo::minTileMemSize(const TileElementSize& tile_size) const noexcept {
  using util::size_t::sum;
  using util::size_t::mul;

  DLAF_ASSERT_HEAVY((tile_size.rows() <= block_size_.rows()));
  DLAF_ASSERT_HEAVY((tile_size.cols() <= block_size_.cols()));

  if (tile_size.isEmpty()) {
    return 0;
  }
  return sum(tile_size.rows(), mul(ld_tile_, tile_size.cols() - 1));
}
}
}
