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
#include "dlaf/tile.h"

namespace dlaf {

template <
    class Tsrc, class Tdst,
    std::enable_if_t<std::is_same<const Tsrc, const Tdst>::value && !std::is_const<Tdst>::value, int> = 0>
void copy(const Tile<Tsrc, Device::CPU>& source, const Tile<Tdst, Device::CPU>& dest) {
  dlaf::tile::lacpy<Tdst>(source, dest);
}

}
