//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/common/index2d.h"

namespace dlaf {
namespace common {

Index2D::Index2D() noexcept : row_(-1), col_(-1) {}

Index2D::Index2D(int row, int col) noexcept : row_(row), col_(col) {}

Index2D::Index2D(const std::array<int, 2> & coords) noexcept : row_(coords[0]), col_(coords[1]) {}

Index2D::operator bool() const noexcept {
  return row_ >= 0 && col_ >= 0;
}

bool Index2D::operator <(const Index2D & boundary) const noexcept {
  return row_ < boundary.row_ && col_ < boundary.col_ && (*this) && boundary;
}

int Index2D::row() const noexcept { return row_; }

int Index2D::col() const noexcept { return col_; }

}
}
