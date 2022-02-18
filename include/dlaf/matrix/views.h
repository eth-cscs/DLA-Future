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

struct IndexHelper {
  IndexHelper(Distribution dist, GlobalElementIndex start_offset)
      : dist_(dist), offset_e_(start_offset) {
    if (dist.size().isEmpty())
      return;

    DLAF_ASSERT(offset_e_.isIn(dist_.size()), offset_e_, dist_.size());

    offset_tile_ = dist_.globalTileIndex(offset_e_);

    tile_begin_ = {
        dist_.nextLocalTileFromGlobalTile<Coord::Row>(offset_tile_.row()),
        dist_.nextLocalTileFromGlobalTile<Coord::Col>(offset_tile_.col()),
    };
  }

  GlobalTileIndex offset() const noexcept {
    return offset_tile_;
  }

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

  LocalTileIndex tile_begin_, tile_end_;
};

struct SubMatrixView : public IndexHelper {
  SubMatrixView(Distribution dist, GlobalElementIndex offset_e) : IndexHelper(dist, offset_e) {
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

struct SubPanelView : public IndexHelper {
  SubPanelView(Distribution dist, GlobalElementIndex offset_e, const SizeType width)
      : IndexHelper(dist, offset_e), cols_(width) {
    if (dist.size().isEmpty())
      return;

    i_sub_offset_ = offset_e_.row() % dist_.blockSize().rows();
    j_sub_offset_ = offset_e_.col() % dist_.blockSize().cols();

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
