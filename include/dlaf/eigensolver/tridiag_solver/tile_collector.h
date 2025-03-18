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

#include <utility>

#include <dlaf/common/range2d.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace dlaf::eigensolver::internal {

/// TileCollector is an helper that, given included boundaries of a range [idx_begin, idx_end),
/// allows to obtain access to all tiles in that range for a matrix.
///
/// The range specified is generally considered a square, i.e. the range is applied to both rows and
/// columns.
/// Instead, if the matrix has a single column of tiles, the range is considered linear over rows.
class TileCollector {
  SizeType idx_begin;
  SizeType idx_end;

  auto iteratorLocal(const matrix::Distribution& dist) const {
    const bool is_col_matrix = dist.nrTiles().cols() == 1;

    const GlobalTileIndex g_begin(idx_begin, is_col_matrix ? 0 : idx_begin);
    const GlobalTileIndex g_end(idx_end, is_col_matrix ? 1 : idx_end);

    const LocalTileIndex begin{
        dist.template nextLocalTileFromGlobalTile<Coord::Row>(g_begin.row()),
        dist.template nextLocalTileFromGlobalTile<Coord::Col>(g_begin.col()),
    };
    const LocalTileIndex end{
        dist.template nextLocalTileFromGlobalTile<Coord::Row>(g_end.row()),
        dist.template nextLocalTileFromGlobalTile<Coord::Col>(g_end.col()),
    };
    const LocalTileSize size = end - begin;

    return std::make_pair(begin, size);
  }

public:
  TileCollector(SizeType i_begin, SizeType i_end) : idx_begin(i_begin), idx_end(i_end) {}

  template <class T, Device D>
  auto read(Matrix<const T, D>& mat) const {
    auto [begin, size] = iteratorLocal(mat.distribution());
    return matrix::selectRead(mat, common::iterate_range2d(begin, size));
  }

  template <class T, Device D>
  auto readwrite(Matrix<T, D>& mat) const {
    auto [begin, size] = iteratorLocal(mat.distribution());
    return matrix::select(mat, common::iterate_range2d(begin, size));
  }
};

}
