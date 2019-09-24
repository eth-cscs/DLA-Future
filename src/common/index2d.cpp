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

Index2D::Index2D(int row, int col) noexcept(false) : row_(row), col_(col) {
  if (!isValid())
    throw std::runtime_error("passed not valid negative indexes");
}

Index2D::Index2D(const std::array<int, 2>& coords) noexcept(false) : Index2D(coords[0], coords[1]) {}

bool Index2D::operator<(const Index2D& boundary) const noexcept {
  return row_ < boundary.row_ && col_ < boundary.col_ && (isValid()) && boundary.isValid();
}

bool Index2D::isValid() const noexcept {
  return row_ >= 0 && col_ >= 0;
}

int Index2D::row() const noexcept {
  return row_;
}

int Index2D::col() const noexcept {
  return col_;
}

Index2D computeCoords(LeadingDimension axis, int index, const std::array<int, 2>& dims) {
  size_t ld_size_index = (axis == LeadingDimension::Row) ? 1 : 0;
  auto leading_size = dims[ld_size_index];

  switch (axis) {
    case LeadingDimension::Row:
      return {index / leading_size, index % leading_size};
    case LeadingDimension::Column:
      return {index % leading_size, index / leading_size};
  }

  throw std::runtime_error("leading dimension specified is not valid");
}

}
}
