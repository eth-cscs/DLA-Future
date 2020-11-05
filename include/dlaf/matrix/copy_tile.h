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

#include <type_traits>

#include "dlaf/lapack_tile.h"
#include "dlaf/matrix/tile.h"

namespace dlaf {

template <class T>
void copy(const matrix::Tile<const T, Device::CPU>& source, const matrix::Tile<T, Device::CPU>& dest) {
  dlaf::tile::lacpy<T>(source, dest);
}

}
