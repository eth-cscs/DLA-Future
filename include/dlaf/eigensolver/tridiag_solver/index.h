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

#include "dlaf/common/callable_object.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/make_sender_algorithm_overloads.h"
#include "dlaf/sender/when_all_lift.h"

namespace dlaf::eigensolver::internal {

inline void initIndexTile(SizeType tile_row, const matrix::Tile<SizeType, Device::CPU>& index) {
  for (SizeType i = 0; i < index.size().rows(); ++i) {
    index(TileElementIndex(i, 0)) = tile_row + i;
  }
}

DLAF_MAKE_CALLABLE_OBJECT(initIndexTile);
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(initIndexTile, initIndexTile_o)

inline void initIndex(SizeType i_begin, SizeType i_end, Matrix<SizeType, Device::CPU>& index) {
  using dlaf::internal::whenAllLift;
  using pika::threads::thread_priority;
  using dlaf::internal::Policy;
  using pika::execution::experimental::start_detached;

  SizeType nb = index.distribution().blockSize().rows();

  for (SizeType i = i_begin; i <= i_end; ++i) {
    GlobalTileIndex tile_idx(i, 0);
    SizeType tile_row = (i - i_begin) * nb;
    whenAllLift(tile_row, index.readwrite_sender(tile_idx)) |
        initIndexTile(Policy<Backend::MC>(thread_priority::normal)) | start_detached();
  }
}

}
