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

/// @file

#include "dlaf/tile.h"

namespace dlaf {

template <class T>
void printElementTile(const Tile<T, Device : CPU>& tile) {
  for (SizeType ii = 0; ii < tile.size().rows(); ++ii) {
    for (SizeType jj = 0; jj < tile.size().cols(); ++jj) {
      std::cout << tile({ii, jj}) << " ";
    }
  }
  std::cout << " " << std::endl;
}

}
