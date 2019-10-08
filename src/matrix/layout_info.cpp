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
#include "dlaf/util_math.h"

namespace dlaf {
namespace matrix {
LayoutInfo::LayoutInfo(const GlobalElementSize& size, const TileElementSize& block_size,
                       SizeType tile_ld, std::size_t tile_offset_row, std::size_t tile_offset_col)
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

  if (size_.isEmpty() == 0) {
    if (ld_tile_ < block_size.rows()) {
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

    if (ld_tile_ < block_size_.rows()) {
      throw std::invalid_argument("Error: Invalid Leading Dimension");
    }
    if (tile_offset_row_ < static_cast<std::size_t>(block_size_.rows())) {
      throw std::invalid_argument("Error: Invalid Tile Row Offset");
    }
    if (tile_offset_row_ < minTileSize(block_size_) &&
        static_cast<std::size_t>(ld_tile_) < sum(tileOffset({nr_tiles_.rows() - 1, 0}), last_rows)) {
      throw std::invalid_argument("Error: Invalid Leading Dimension & Tile Row Offset combination");
    }
    if (tile_offset_col_ <
        tileOffset({nr_tiles_.rows() - 1, 0}) + minTileSize({last_rows, block_size_.cols()})) {
      throw std::invalid_argument("Error: Invalid Tile Col Offset");
    }
  }
}

std::size_t LayoutInfo::minMemSize() const noexcept {
  if (size_.isEmpty()) {
    return 0;
  }

  SizeType last_rows = size_.rows() - block_size_.rows() * (nr_tiles_.rows() - 1);
  SizeType last_cols = size_.cols() - block_size_.cols() * (nr_tiles_.cols() - 1);

  return tileOffset({nr_tiles_.rows() - 1, nr_tiles_.cols() - 1}) + minTileSize({last_rows, last_cols});
}

std::size_t LayoutInfo::minTileSize(const TileElementSize& size) const noexcept {
  using util::size_t::sum;
  using util::size_t::mul;

  if (size_.isEmpty()) {
    return 0;
  }

  return sum(size.rows(), mul(ld_tile_, size.cols() - 1));
}
}
}
