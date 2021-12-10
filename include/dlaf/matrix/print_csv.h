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

#include "dlaf/common/index2d.h"
#include "dlaf/common/range2d.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {

namespace format {
struct csv {};
}

namespace matrix {

/// Print a tile in csv format to standard output
template <class T>
void print(format::csv, const Tile<const T, Device::CPU>& tile, std::ostream& os = std::cout) {
  for (SizeType ii = 0; ii < tile.size().rows(); ++ii) {
    for (SizeType jj = 0; jj < tile.size().cols(); ++jj) {
      os << std::setprecision(5) << tile({ii, jj}) << ",";
    }
    os << std::endl;
  }
}

/// Print a matrix in csv format to standard output
///
/// If the matrix is distributed, the basic matrix info will be printed  using operator << of the Matrix.
template <class T>
void print(format::csv, std::string sym, Matrix<const T, Device::CPU>& mat,
           std::ostream& os = std::cout) {
  if (!local_matrix(mat)) {
    os << mat << std::endl;
    return;
  }

  SizeType nrow = mat.size().rows();
  SizeType ncol = mat.size().cols();
  SizeType blockrow = mat.blockSize().rows();
  SizeType blockcol = mat.blockSize().cols();

  os << sym << std::endl;

  for (SizeType irow = 0; irow < nrow; ++irow) {
    SizeType tilerow = irow / blockrow;
    SizeType elrow = irow % blockrow;

    for (SizeType icol = 0; icol < ncol; ++icol) {
      SizeType tilecol = icol / blockcol;
      SizeType elcol = icol % blockcol;

      const LocalTileIndex idx = {tilerow, tilecol};
      auto& tile = mat.read(idx).get();

      os << tile({elrow, elcol}) << ",";
    }
    os << std::endl;
  }
}
}
}
