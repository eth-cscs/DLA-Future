//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
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

/// ColMajorLayout contains the information about how the elements of the local part of the matrix are
/// stored in memory.
class ColMajorLayout {
public:
  /// Construct a column major layout of a matrix eith distribution @p distribution
  ///
  /// @pre leading_dimension >= max(1, distribution.local_size().rows())
  ColMajorLayout(const Distribution& distribution, SizeType leading_dimension)
      : dist_(distribution), ld_(leading_dimension) {
    if (dist_.local_size().isEmpty()) {
      DLAF_ASSERT(ld_ >= 1, ld_);
    }
    else {
      DLAF_ASSERT(ld_ >= dist_.local_size().rows(), ld_, dist_.local_size());
    }
    DLAF_ASSERT(dist_.offset() == (GlobalElementIndex{0, 0}), dist_.offset());
  }

  bool operator==(const ColMajorLayout& rhs) const noexcept {
    return dist_ == rhs.dist_ && ld_ == rhs.ld_;
  }

  bool operator!=(const ColMajorLayout& rhs) const noexcept {
    return !operator==(rhs);
  }

  /// Returns the minimum number of elements that are needed to fit a matrix with the given layout.
  SizeType min_mem_size() const noexcept {
    return min_tile_mem_size(size().rows(), size().cols());
  }

  /// Returns the position of the first element of the @p index tile.
  ///
  /// @pre index.isIn(nr_tiles()).
  SizeType tile_offset(const LocalTileIndex& index) const noexcept {
    DLAF_ASSERT_HEAVY(index.isIn(nr_tiles()), index, nr_tiles());
    SizeType i = dist_.local_element_from_local_tile_and_tile_element<Coord::Row>(index.row(), 0);
    SizeType j = dist_.local_element_from_local_tile_and_tile_element<Coord::Col>(index.col(), 0);
    return i + j * ld_;
  }

  /// Returns the size the @p index tile.
  ///
  /// @pre index.isIn(nr_tiles()).
  TileElementSize tile_size_of(const LocalTileIndex& index) const noexcept {
    return dist_.tile_size_of(index);
  }

  /// Returns the minimum number of elements that are needed for the @p index tile.
  ///
  /// @pre index.isIn(nr_tiles()).
  SizeType ld_tile(const LocalTileIndex&) const noexcept {
    return ld_;
  }

  /// Returns the minimum number of elements that are needed for the @p index tile.
  ///
  /// @pre index.isIn(nr_tiles()).
  SizeType min_tile_mem_size(const LocalTileIndex& index) const noexcept {
    DLAF_ASSERT_HEAVY(index.isIn(nr_tiles()), index, nr_tiles());
    const TileElementSize tile_size = tile_size_of(index);

    return min_tile_mem_size(tile_size.rows(), tile_size.cols());
  }

  const LocalElementSize& size() const noexcept {
    return dist_.local_size();
  }

  const LocalTileSize& nr_tiles() const noexcept {
    return dist_.local_nr_tiles();
  }

  const TileElementSize& tile_size() const noexcept {
    return dist_.tile_size();
  }

  const Distribution& distribution() const noexcept {
    return dist_;
  }

  constexpr static MatrixAllocation allocation() noexcept {
    return MatrixAllocation::ColMajor;
  }

private:
  SizeType min_tile_mem_size(const SizeType m, SizeType n) const noexcept {
    DLAF_ASSERT_HEAVY(m >= 0, m);
    DLAF_ASSERT_HEAVY(n >= 0, n);

    if (m == 0 || n == 0) {
      return 0;
    }
    return m + ld_ * (n - 1);
  }

  Distribution dist_;
  SizeType ld_;
};
}
