//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
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
                       SizeType tile_offset_row, SizeType tile_offset_col)
    : size_(size), nr_tiles_(0, 0), block_size_(block_size), ld_tile_(tile_ld),
      tile_offset_row_(tile_offset_row), tile_offset_col_(tile_offset_col) {
  using dlaf::util::ceilDiv;

  DLAF_ASSERT(size_.isValid(), size_);
  DLAF_ASSERT(!block_size_.isEmpty(), block_size_);

  nr_tiles_ = {ceilDiv(size_.rows(), block_size_.rows()), ceilDiv(size_.cols(), block_size_.cols())};

  if (size_.isEmpty()) {
    DLAF_ASSERT(ld_tile_ >= 1, ld_tile_);
    DLAF_ASSERT(tile_offset_row_ >= 1, tile_offset_row);
    DLAF_ASSERT(tile_offset_col_ >= 1, tile_offset_col);
  }
  else {
    SizeType last_rows = size_.rows() - block_size_.rows() * (nr_tiles_.rows() - 1);

    SizeType max_rows_tiles = std::min(size_.rows(), block_size_.rows());

    DLAF_ASSERT(ld_tile_ >= max_rows_tiles, ld_tile_, max_rows_tiles);
    DLAF_ASSERT(tile_offset_row_ >= max_rows_tiles, tile_offset_row, max_rows_tiles);
    DLAF_ASSERT(tile_offset_row_ >= minTileMemSize(block_size_) ||
                    ld_tile_ >= tileOffset({nr_tiles_.rows() - 1, 0}) + last_rows,
                tile_offset_row_, minTileMemSize(block_size_), ld_tile_,
                tileOffset({nr_tiles_.rows() - 1, 0}), last_rows);
    DLAF_ASSERT(tile_offset_col_ >= tileOffset({nr_tiles_.rows() - 1, 0}) +
                                        minTileMemSize(LocalTileIndex(nr_tiles_.rows() - 1, 0)),
                tile_offset_col_,
                tileOffset({nr_tiles_.rows() - 1, 0}) +
                    minTileMemSize(LocalTileIndex(nr_tiles_.rows() - 1, 0)));
  }
}

SizeType LayoutInfo::minMemSize() const noexcept {
  if (size_.isEmpty()) {
    return 0;
  }

  LocalTileIndex index_last(nr_tiles_.rows() - 1, nr_tiles_.cols() - 1);
  return tileOffset(index_last) + minTileMemSize(index_last);
}

SizeType LayoutInfo::minTileMemSize(const TileElementSize& tile_size) const noexcept {
  DLAF_ASSERT_HEAVY(tile_size.rows() <= block_size_.rows(), tile_size, block_size_);
  DLAF_ASSERT_HEAVY(tile_size.cols() <= block_size_.cols(), tile_size, block_size_);

  if (tile_size.isEmpty()) {
    return 0;
  }
  return tile_size.rows() + ld_tile_ * (tile_size.cols() - 1);
}
}
}
