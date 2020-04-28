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

namespace dlaf {

/// Copy values from another matrix
///
/// Given a matrix with the same geometries and distribution, this function submits tasks that will
/// perform the copy of each tile
template <template <class, Device> class MatrixTypeSrc, template <class, Device> class MatrixTypeDst,
          class Tsrc, class Tdst>
void copy(MatrixTypeSrc<Tsrc, Device::CPU>& source, MatrixTypeDst<Tdst, Device::CPU>& dest) {
  static_assert(std::is_same<const Tsrc, const Tdst>::value,
                "Source and destination matrix should have the same type");
  static_assert(!std::is_const<Tdst>::value, "Destination matrix cannot be const");

  const auto& distribution = source.distribution();

  // TODO check same size and blocksize
  // TODO check equally distributed

  const SizeType local_tile_rows = distribution.localNrTiles().rows();
  const SizeType local_tile_cols = distribution.localNrTiles().cols();

  for (SizeType j = 0; j < local_tile_cols; ++j)
    for (SizeType i = 0; i < local_tile_rows; ++i)
      hpx::dataflow(hpx::util::unwrapping(dlaf::copy<Tsrc, Tdst>), source.read(LocalTileIndex(i, j)),
                    dest(LocalTileIndex(i, j)));
}

}
