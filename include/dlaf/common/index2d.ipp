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

namespace internal {
template <typename index_t, class Tag>
basic_coords<index_t, Tag>::basic_coords() noexcept : row_(-1), col_(-1) {}

template <typename index_t, class Tag>
basic_coords<index_t, Tag>::basic_coords(index_t row, index_t col) noexcept(false)
    : row_(row), col_(col) {
  if (!isValid())
    throw std::runtime_error("passed not valid negative indexes");
}

template <typename index_t, class Tag>
basic_coords<index_t, Tag>::basic_coords(const std::array<index_t, 2>& coords) noexcept(false)
    : basic_coords(coords[0], coords[1]) {}

template <typename index_t, class Tag>
bool basic_coords<index_t, Tag>::isValid() const noexcept {
  return row_ >= 0 && col_ >= 0;
}

template <typename index_t, class Tag>
index_t basic_coords<index_t, Tag>::row() const noexcept {
  return row_;
}

template <typename index_t, class Tag>
index_t basic_coords<index_t, Tag>::col() const noexcept {
  return col_;
}
}

template <typename index_t, class Tag>
bool Index2D<index_t, Tag>::isIn(const Size2D<index_t, Tag>& boundary) const noexcept {
  return this->row_ < boundary.row_ && this->col_ < boundary.col_ && (this->isValid()) &&
         boundary.isValid();
}

template <typename index_t, typename linear_t, class Tag>
void computeCoords(Ordering ordering, linear_t index, const std::array<index_t, 2>& dims,
                   internal::basic_coords<index_t, Tag>& coords) {
  std::size_t ld_size_index = (ordering == Ordering::RowMajor) ? 1 : 0;
  auto leading_size = dims[ld_size_index];

  switch (ordering) {
    case Ordering::RowMajor:
      coords = {static_cast<index_t>(index / leading_size), static_cast<index_t>(index % leading_size)};
      return;
    case Ordering::ColumnMajor:
      coords = {static_cast<index_t>(index % leading_size), static_cast<index_t>(index / leading_size)};
      return;
  }

  throw std::runtime_error("leading dimension specified is not valid");
}

}
}
