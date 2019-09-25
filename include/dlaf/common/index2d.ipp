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

template <typename index_t>
Index2D<index_t>::Index2D() noexcept : row_(-1), col_(-1) {}

template <typename index_t>
Index2D<index_t>::Index2D(index_t row, index_t col) noexcept(false) : row_(row), col_(col) {
  if (!isValid())
    throw std::runtime_error("passed not valid negative indexes");
}

template <typename index_t>
Index2D<index_t>::Index2D(const std::array<index_t, 2>& coords) noexcept(false)
    : Index2D(coords[0], coords[1]) {}

template <typename index_t>
bool Index2D<index_t>::operator<(const Index2D& boundary) const noexcept {
  return row_ < boundary.row_ && col_ < boundary.col_ && (isValid()) && boundary.isValid();
}

template <typename index_t>
bool Index2D<index_t>::isValid() const noexcept {
  return row_ >= 0 && col_ >= 0;
}

template <typename index_t>
index_t Index2D<index_t>::row() const noexcept {
  return row_;
}

template <typename index_t>
index_t Index2D<index_t>::col() const noexcept {
  return col_;
}

template <typename index_t, typename linear_t>
Index2D<index_t> computeCoords(Ordering ordering, linear_t index,
                               const std::array<index_t, 2>& dims) {
  std::size_t ld_size_index = (ordering == Ordering::RowMajor) ? 1 : 0;
  auto leading_size = dims[ld_size_index];

  switch (ordering) {
    case Ordering::RowMajor:
      return {static_cast<index_t>(index / leading_size), static_cast<index_t>(index % leading_size)};
    case Ordering::ColumnMajor:
      return {static_cast<index_t>(index % leading_size), static_cast<index_t>(index / leading_size)};
  }

  throw std::runtime_error("leading dimension specified is not valid");
}

}
}
