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

#include "dlaf/common/range2d.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf::eigensolver::internal {

/// TileCollector is an helper that, given included boundaries of a range [idx_begin, idx_last],
/// allows to obtain access to all tiles in that range for a matrix.
///
/// The range specified is generally considered a square, i.e. the range is applied to both rows and
/// columns.
/// Instead, if the matrix has a single column of tiles, the range is considered linear over rows.
class TileCollector {
  SizeType idx_begin;
  SizeType idx_last;

  auto iteratorLocal(const matrix::Distribution& dist) const {
    const bool is_col_matrix = dist.nrTiles().cols() == 1;

    const GlobalTileIndex g_begin(idx_begin, is_col_matrix ? 0 : idx_begin);
    const GlobalTileIndex g_end(idx_last + 1, is_col_matrix ? 1 : idx_last + 1);

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
  TileCollector(SizeType i_begin, SizeType i_last) : idx_begin(i_begin), idx_last(i_last) {}

  template <class T, Device D>
  auto read(Matrix<const T, D>& mat) const {
    auto [begin, size] = iteratorLocal(mat.distribution());
    return matrix::util::collectReadTiles(begin, size, mat);
  }

  template <class T, Device D>
  auto readwrite(Matrix<T, D>& mat) const {
    auto [begin, size] = iteratorLocal(mat.distribution());
    return matrix::util::collectReadWriteTiles(begin, size, mat);
  }
};

}
