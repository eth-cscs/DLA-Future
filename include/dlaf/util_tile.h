//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

/// @file

#pragma once

#include "dlaf/tile.h"
#include "dlaf/common/buffer.h"

namespace dlaf {

/// @brief Create a common::Buffer from a Tile
template <class T, Device device>
auto create_buffer(const dlaf::Tile<T, device> & tile) {
  return dlaf::common::Buffer<T*>(tile.ptr({0, 0}), tile.size().cols(), tile.size().rows(), tile.ld());
}

}
