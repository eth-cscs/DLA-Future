//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <pika/execution.hpp>

#include "dlaf/common/range2d.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"

namespace dlaf::matrix {

/// Copy subset of local tiles from @p source in the range @p idx_source_begin, @p sz to local subset of
/// tiles in @p dest in the range @p idx_dest_begin, @p sz.
///
/// Note: If @p source and @p dest are the same matrix and iteration ranges overlap partially, the result
///       of the copy will be incorrect.
///
template <class T, Device Source, Device Destination>
void copy(LocalTileSize sz, LocalTileIndex idx_source_begin, Matrix<const T, Source>& source,
          LocalTileIndex idx_dest_begin, Matrix<T, Destination>& dest) {
  // If @p source and @p dest is the same matrix and the iteration range fully overlaps, return.
  if constexpr (Source == Destination) {
    if (idx_source_begin == idx_dest_begin && &source == &dest)
      return;
  }

  // Given that `sz` is the same for both `source` and `dest` it is sufficient to only check if the local
  // length of the copied region is the same.
  DLAF_ASSERT(source.distribution().localElementDistanceFromLocalTile(idx_source_begin,
                                                                      idx_source_begin + sz) ==
                  dest.distribution().localElementDistanceFromLocalTile(idx_dest_begin,
                                                                        idx_dest_begin + sz),
              source, dest);

  namespace ex = pika::execution::experimental;
  for (auto idx_dest : common::iterate_range2d(idx_dest_begin, sz)) {
    LocalTileIndex idx_source = idx_source_begin + (idx_dest - idx_dest_begin);
    ex::start_detached(ex::when_all(source.read(idx_source), dest.readwrite(idx_dest)) |
                       copy(dlaf::internal::Policy<internal::CopyBackend_v<Source, Destination>>{}));
  }
}

/// \overload copy()
///
/// This overload makes sure that both @p source and @p dest local tiles start @p idx_begin.
///
template <class T, Device Source, Device Destination>
void copy(LocalTileIndex idx_begin, LocalTileSize sz, Matrix<const T, Source>& source,
          Matrix<T, Destination>& dest) {
  copy(sz, idx_begin, source, idx_begin, dest);
}

/// \overload copy()
///
/// This overload makes sure that all local tiles of @p source are copied to @p dest starting at tile (0, 0).
///
template <class T, Device Source, Device Destination>
void copy(Matrix<const T, Source>& source, Matrix<T, Destination>& dest) {
  copy(source.distribution().localNrTiles(), LocalTileIndex(0, 0), source, LocalTileIndex(0, 0), dest);
}

}
