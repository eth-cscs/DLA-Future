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

#include "dlaf/matrix.h"

namespace dlaf {

template <class T>
void printElements(Matrix<T, Device::CPU>& mat) {
  SizeType nrow = mat.size().rows();
  SizeType ncol = mat.size().cols();
  SizeType blockrow = mat.blockSize().rows();
  SizeType blockcol = mat.blockSize().cols();

  for (SizeType irow = 0; irow < nrow; ++irow) {
    SizeType tilerow = irow / blockrow;
    SizeType elrow = irow % blockrow;

    for (SizeType icol = 0; icol < ncol; ++icol) {
      SizeType tilecol = icol / blockcol;
      SizeType elcol = icol % blockcol;

      const LocalTileIndex idx = {tilerow, tilecol};
      auto tile = mat(idx).get();

      std::cout << tile({elrow, elcol}) << " ";
    }
    std::cout << " " << std::endl;
  }
  std::cout << " " << std::endl;
}

}
