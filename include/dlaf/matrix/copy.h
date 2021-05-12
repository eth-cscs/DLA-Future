//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <hpx/include/util.hpp>
#include <hpx/local/future.hpp>

#include "dlaf/executors.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace matrix {

/// Copy values from another matrix.
///
/// Given a matrix with the same geometries and distribution, this function submits tasks that will
/// perform the copy of each tile.
template <class T, Device Source, Device Destination>
void copy(Matrix<const T, Source>& source, Matrix<T, Destination>& dest) {
  const auto& distribution = source.distribution();

  DLAF_ASSERT(matrix::equal_size(source, dest), source, dest);
  DLAF_ASSERT(matrix::equal_blocksize(source, dest), source, dest);
  DLAF_ASSERT(matrix::equal_distributions(source, dest), source, dest);

  const SizeType local_tile_rows = distribution.localNrTiles().rows();
  const SizeType local_tile_cols = distribution.localNrTiles().cols();

  for (SizeType j = 0; j < local_tile_cols; ++j)
    for (SizeType i = 0; i < local_tile_rows; ++i)
      hpx::dataflow(dlaf::getCopyExecutor<Source, Destination>(), unwrapExtendTiles(copy_o),
                    source.read(LocalTileIndex(i, j)), dest(LocalTileIndex(i, j)));
}
}
}
