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
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/util_math.h"

namespace dlaf {

class MatrixBase {
public:
  MatrixBase() noexcept;

  MatrixBase(GlobalElementSize size, TileElementSize block_size);

  MatrixBase(matrix::LayoutInfo layout) noexcept;

  MatrixBase(const MatrixBase& rhs) = default;

  MatrixBase(MatrixBase&& rhs) noexcept;

  MatrixBase& operator=(const MatrixBase& rhs) = default;

  MatrixBase& operator=(MatrixBase&& rhs) noexcept;

  bool operator==(const MatrixBase& rhs) const noexcept {
    return size_ == rhs.size_ && nr_tiles_ == rhs.nr_tiles_ && block_size_ == rhs.block_size_;
  }

  bool operator!=(const MatrixBase& rhs) const noexcept {
    return !operator==(rhs);
  }

  GlobalElementSize size() const noexcept {
    return size_;
  }

  LocalTileSize nrTiles() const noexcept {
    return nr_tiles_;
  }

  TileElementSize blockSize() const noexcept {
    return block_size_;
  }

private:
  GlobalElementSize size_;
  LocalTileSize nr_tiles_;
  TileElementSize block_size_;

  // id and size rank_src in 2D grid.
};
}
