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

#include "dlaf/common/index2d.h"
#include "dlaf/common/range2d.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"

namespace dlaf {

namespace format {
struct human {};
}

namespace matrix {

namespace internal {

/// Print a matrix in a human readable format to standard output
// TODO: now is printed to standard output, would it be better to save it into a file?
template <class T>
void print(format::human, Matrix<T, Device::CPU>& mat) {
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
      auto& tile = mat.read(idx).get();

      std::cout << tile({elrow, elcol}) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

/// Print a tile in a human readable format to standard output
template <class T>
void print(format::human, const Tile<T, Device::CPU>& tile) {
  for (SizeType ii = 0; ii < tile.size().rows(); ++ii) {
    for (SizeType jj = 0; jj < tile.size().cols(); ++jj) {
      std::cout << std::setprecision(5) << tile({ii, jj}) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

}
}
}
