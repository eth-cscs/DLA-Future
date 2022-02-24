//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <ios>
#include "dlaf/common/assert.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/range2d.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/tile.h"

namespace dlaf::matrix {

namespace internal {
/// Helper for accessing a Matrix whose top-left offset is not necessarily tile-aligned.
///
/// It is defined in terms of global reference system and, in addition to providing an iterator to access
/// all local tiles part of the view, it computes the correct SubTileSpec to use for accessing the
/// right part of the tile for each LocalTileIndex in its range.
struct View {
  /// Create a view, for a Distribution @p dist, whose top-left corner is @p offset_e
  ///
  /// This internal helper does not fully initializes the range. Indeed just @p tile_begin_ gets
  /// initialized according to the top-left corner specified, while @p tile_end_ is left to the
  /// deriving implementation to be initialized according to their needs.
  ///
  /// If an empty distribution is passed, the range is initialized to be the empty interval starting and
  /// ending in LocalTileIndex(0, 0).
  ///
  /// @param dist is the distribution of the matrix on which the view is applied
  /// @param offset_e is the top left corner where the view starts in the global matrix
  /// @pre not dist.size().isEmpty() => start_offset.isIn(dist.size())
  View(Distribution dist, GlobalElementIndex start_offset) : dist_(dist), offset_e_(start_offset) {
    if (dist.size().isEmpty())
      return;

    DLAF_ASSERT(offset_e_.isIn(dist_.size()), offset_e_, dist_.size());

    offset_tile_ = dist_.globalTileIndex(offset_e_);

    tile_begin_ = {
        dist_.nextLocalTileFromGlobalTile<Coord::Row>(offset_tile_.row()),
        dist_.nextLocalTileFromGlobalTile<Coord::Col>(offset_tile_.col()),
    };
  }

  /// Return the top left corner
  GlobalTileIndex offset() const noexcept {
    return offset_tile_;
  }

  /// Return a Range2D that gives access to all local tiles part of the View
  auto iteratorLocal() const noexcept {
    return common::iterate_range2d(tile_begin_, tile_end_);
  }

  // TODO this is used for determing first, but it's not a solution
  LocalTileIndex begin() const noexcept {
    return tile_begin_;
  }

protected:
  Distribution dist_;
  GlobalElementIndex offset_e_;
  GlobalTileIndex offset_tile_;

  LocalTileIndex tile_begin_ = {0, 0}, tile_end_ = {0, 0};
};
}

struct SubMatrixView : public internal::View {
  /// Create a SubMatrixView, for a Distribution @p dist, whose top-left corner is @p offset_e, and the
  /// bottom-right corner is implicitly set at the bottom-right limit of the global matrix.
  ///
  /// If an empty distribution is passed, the range is initialized to be the empty interval starting and
  /// ending in LocalTileIndex(0, 0).
  ///
  /// @param dist is the distribution of the matrix on which the view is applied
  /// @param offset_e is the top left corner where the view starts in the global matrix
  SubMatrixView(Distribution dist, GlobalElementIndex offset_e) : View(dist, offset_e) {
    if (dist.size().isEmpty())
      return;

    tile_end_ = indexFromOrigin(dist_.localNrTiles());

    has_subtile_ = dist_.rankGlobalTile<Coord::Row>(offset().row()) == dist_.rankIndex().row() ||
                   dist_.rankGlobalTile<Coord::Col>(offset().col()) == dist_.rankIndex().col();
  }

  matrix::SubTileSpec operator()(const LocalTileIndex& index) const {
    DLAF_ASSERT_MODERATE(isIndexInRange(index, iteratorLocal()), index, tile_begin_, tile_end_);

    SizeType i_sub = 0, j_sub = 0;

    if (has_subtile_) {
      const auto ij = dist_.globalTileIndex(index);

      if (ij.row() == offset().row())
        i_sub += dist_.tileElementFromGlobalElement<Coord::Row>(offset_e_.row());
      if (ij.col() == offset().col())
        j_sub += dist_.tileElementFromGlobalElement<Coord::Col>(offset_e_.col());
    }
    const TileElementIndex offset_sub(i_sub, j_sub);

    const TileElementSize size_tile = dist_.tileSize(dist_.globalTileIndex(index));
    const TileElementSize size_sub(size_tile.rows() - offset_sub.row(),
                                   size_tile.cols() - offset_sub.col());
    return {offset_sub, size_sub};
  }

private:
  bool has_subtile_;
};

struct SubPanelView : public internal::View {
  /// Create a SubPanelView, for a Distribution @p dist, whose top-left corner is @p offset_e, and the
  /// bottom-right corner is set at the bottom row and @p ncols columns on the right in the global matrix.
  ///
  /// It is constrained to be 1 column wide in terms of tiles.
  ///
  /// If an empty distribution is passed, the range is initialized to be the empty interval starting and
  /// ending in LocalTileIndex(0, 0).
  ///
  /// @param dist is the distribution of the matrix on which the view is applied
  /// @param offset_e is the top left corner where the view starts in the global matrix
  /// @param ncols is the number of columns composing this panel
  ///
  /// @pre offset_e.col() + ncols <= dist.blockSize().cols()
  SubPanelView(Distribution dist, GlobalElementIndex offset_e, const SizeType ncols)
      : View(dist, offset_e), cols_(ncols) {
    i_sub_offset_ = offset_e_.row() % dist_.blockSize().rows();
    j_sub_offset_ = offset_e_.col() % dist_.blockSize().cols();

    DLAF_ASSERT(j_sub_offset_ + cols_ <= dist_.blockSize().cols(), j_sub_offset_ + cols_,
                dist_.blockSize().cols());

    if (dist.size().isEmpty())
      return;

    tile_end_ = LocalTileIndex(dist_.localNrTiles().rows(), tile_begin_.col() + 1);

    has_top_tile_ = dist_.rankGlobalTile<Coord::Row>(offset().row()) == dist_.rankIndex().row();
  }

  matrix::SubTileSpec operator()(const LocalTileIndex& index) const {
    DLAF_ASSERT_MODERATE(isIndexInRange(index, iteratorLocal()), index, tile_begin_, tile_end_);

    const TileElementIndex offset_sub((has_top_tile_ && index == tile_begin_) ? i_sub_offset_ : 0,
                                      j_sub_offset_);

    const TileElementSize size_tile = dist_.tileSize(dist_.globalTileIndex(index));
    const TileElementSize size_sub(size_tile.rows() - offset_sub.row(),
                                   std::min(cols_, size_tile.cols() - offset_sub.col()));
    return {offset_sub, size_sub};
  }

  /// Return the number of columns of elements part of this panel view
  SizeType cols() const noexcept {
    return cols_;
  }

private:
  ///> Maximum width for the panel (it can be less in case there are not enough elements in the matrix)
  SizeType cols_;
  SizeType i_sub_offset_, j_sub_offset_;

  bool has_top_tile_;
};

}
