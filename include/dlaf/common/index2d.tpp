//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

namespace dlaf {
namespace common {

namespace internal {
template <typename IndexT>
basic_coords<IndexT>::basic_coords(IndexT row, IndexT col) noexcept : row_(row), col_(col) {}

template <typename IndexT>
basic_coords<IndexT>::basic_coords(const std::array<IndexT, 2>& coords) noexcept
    : basic_coords(coords[0], coords[1]) {}
}

template <typename IndexT, class Tag>
Index2D<IndexT, Tag>::Index2D() noexcept : internal::basic_coords<IndexT>(-1, -1) {}

template <typename IndexT, class Tag>
Index2D<IndexT, Tag>::Index2D(IndexT row, IndexT col) : internal::basic_coords<IndexT>(row, col) {
  if (!internal::basic_coords<IndexT>::isValid())
    throw std::invalid_argument("indices are not valid (negative).");
}

template <typename IndexT, class Tag>
Index2D<IndexT, Tag>::Index2D(const std::array<IndexT, 2>& coords) : Index2D(coords[0], coords[1]) {}

template <typename IndexT, class Tag>
bool Index2D<IndexT, Tag>::isIn(const Size2D<IndexT, Tag>& boundary) const noexcept {
  return this->row() < boundary.rows() && this->col() < boundary.cols() && (this->isValid()) &&
         boundary.isValid();
}

/// Compute Index2D of a linear index inside a 2D grid with specified size and ordering
template <class Index2DType, typename LinearIndexT>
Index2DType computeCoords(Ordering ordering, LinearIndexT index,
                          const std::array<typename Index2DType::IndexType, 2>& dims) {
  using IndexType = typename Index2DType::IndexType;

  switch (ordering) {
    case Ordering::RowMajor:
      return {static_cast<IndexType>(index / dims[1]), static_cast<IndexType>(index % dims[1])};
    case Ordering::ColumnMajor:
      return {static_cast<IndexType>(index % dims[0]), static_cast<IndexType>(index / dims[0])};
    default:
      return {};
  }
}

}
}
