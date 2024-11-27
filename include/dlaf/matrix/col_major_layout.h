//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <dlaf/common/assert.h>
#include <dlaf/matrix/allocation.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/index.h>
#include <dlaf/util_math.h>

namespace dlaf::matrix {
/*

/// ColMajorLayout contains the information about how the elements of a matrix are stored in memory.

class ColMajorLayout {
  public:
  /// Construct a column major layout of a matrix eith distribution @p distribution
  ///
  /// @pre leading_dimension >= max(1, distribution.local_size().rows())
  ColMajorLayout(const Distribution& distribution, SizeType leading_dimension) : dist_(distribution), ld_(leading_dimension) {
    DLAF_ASSERT(ld_ >= std::max(1, distribution.local_size().rows()), ld_, distribution.local_size().rows());
  }

  bool operator==(const ColMajorLayout& rhs) const noexcept {
    return dist_ == rhs.dist_ &&
        ld_ == rhs.ld_;
  }

  bool operator!=(const ColMajorLayout& rhs) const noexcept {
    return !operator==(rhs);
  }

  /// Returns the minimum number of elements that are needed to fit a matrix with the given layout.
  SizeType min_mem_size() const noexcept {
    if (dist_.size().isEmpty()) {
      return 0;
    }

    LocalTileIndex index_last(nr_tiles_.rows() - 1, nr_tiles_.cols() - 1);
    return size_.rows() + ld_ * (size_.cols() - 1);
  }

  /// Returns the position of the first element of the @p index tile.
  ///
  /// @pre index.isIn(nr_tiles_).
  SizeType tile_offset(const LocalTileIndex& index) const noexcept {
    DLAF_ASSERT_HEAVY(index.isIn(dist_.local_nr_tiles()), index, dist_.local_nr_tiles());
    return index.row() * tile_offset_row_ + index.col() * tile_offset_col_;
  }

  /// Returns the size the @p index tile.
  ///
  /// @pre index.isIn(nr_tiles_).
  TileElementSize tile_size_of(const LocalTileIndex& index) const noexcept {
    DLAF_ASSERT_HEAVY(index.isIn(dist_.local_nr_tiles()), index, dist_.local_nr_tiles());
    SizeType m = std::min(tile_size_.rows(), size_.rows() - index.row() * tile_size_.rows());
    SizeType n = std::min(tile_size_.cols(), size_.cols() - index.col() * tile_size_.cols());
    return {m, n};
  }

  /// Returns the minimum number of elements that are needed for the @p index tile.
  ///
  /// @pre index.isIn(nr_tiles_).
  SizeType ld_tile(const LocalTileIndex&) const noexcept {
    DLAF_ASSERT_HEAVY(index.isIn(nr_tiles_), index, nr_tiles_);
    return ld_;
  }

  /// Returns the minimum number of elements that are needed for the @p index tile.
  ///
  /// @pre index.isIn(nr_tiles_).
  SizeType min_tile_mem_size(const LocalTileIndex& index) const noexcept {
    DLAF_ASSERT_HEAVY(index.isIn(nr_tiles_), index, nr_tiles_);
    return min_tile_mem_size(tile_size_of(index));
  }

  const LocalElementSize& size() const noexcept {
    return size_;
  }

  const LocalTileSize& nr_tiles() const noexcept {
    return nr_tiles_;
  }

  const TileElementSize& tile_size() const noexcept {
    return tile_size_;
  }

  constexpr static MatrixAllocation allocation_type() noexcept {
    return MatrixAllocation::RowMajor;
  }

  private:
  SizeType min_tile_mem_size(const TileElementSize& tile_size) const noexcept {
    DLAF_ASSERT_HEAVY(tile_size.rows() <= tile_size_.rows(), tile_size, tile_size_);
    DLAF_ASSERT_HEAVY(tile_size.cols() <= tile_size_.cols(), tile_size, tile_size_);

    if (tile_size.isEmpty()) {
      return 0;
    }
    return tile_size.rows() + ld_ * (tile_size.cols() - 1);
  }

  Distribution dist_;
  SizeType ld_;
};

/// Returns ColMajorLayout for a local column major matrix.
inline ColMajorLayout col_major_layout(const LocalElementSize& size, const TileElementSize& tile_size,
                                       SizeType ld) {
  return ColMajorLayout(size, tile_size, ld, tile_size.rows(), tile_size.cols() * ld);
}
/// Returns ColMajorLayout for a distributed column major matrix.
inline ColMajorLayout col_major_layout(const matrix::Distribution& distribution, SizeType ld) {
  return colMajorLayout(distribution.local_size(), distribution.tile_size(), ld);
}*/
}
