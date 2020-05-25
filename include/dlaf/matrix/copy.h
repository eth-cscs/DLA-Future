//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <hpx/hpx.hpp>

#include "dlaf/matrix/copy_tile.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {

/// Copy values from another matrix
///
/// Given a matrix with the same geometries and distribution, this function submits tasks that will
/// perform the copy of each tile
template <template <class, Device> class MatrixTypeSrc, template <class, Device> class MatrixTypeDst,
          class T, Device device>
void copy(MatrixTypeSrc<const T, device>& source, MatrixTypeDst<T, device>& dest) {
  const auto& distribution = source.distribution();

  DLAF_ASSERT_SIZE_EQ(source, dest);
  DLAF_ASSERT_BLOCKSIZE_EQ(source, dest);
  DLAF_ASSERT_DISTRIBUTED_EQ(source, dest);

  const SizeType local_tile_rows = distribution.localNrTiles().rows();
  const SizeType local_tile_cols = distribution.localNrTiles().cols();

  for (SizeType j = 0; j < local_tile_cols; ++j)
    for (SizeType i = 0; i < local_tile_rows; ++i)
      hpx::dataflow(hpx::util::unwrapping(dlaf::copy<T>), source.read(LocalTileIndex(i, j)),
                    dest(LocalTileIndex(i, j)));
}

}
