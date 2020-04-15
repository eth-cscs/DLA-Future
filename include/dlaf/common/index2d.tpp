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
template <class IndexType, class LinearIndexT, class IndexTag>
Index2D<IndexType, IndexTag> computeCoords(Ordering ordering, LinearIndexT index,
                                           const Size2D<IndexType, IndexTag>& dims) {
  switch (ordering) {
    case Ordering::RowMajor:
      return {static_cast<IndexType>(index / dims.cols()), static_cast<IndexType>(index % dims.cols())};
    case Ordering::ColumnMajor:
      return {static_cast<IndexType>(index % dims.rows()), static_cast<IndexType>(index / dims.rows())};
  }

  throw std::invalid_argument("Ordering specified is not valid.");
}

/// Compute linear index of an Index2D
///
/// @return -1 if given index is outside the grid size, otherwise the linear index (w.r.t specified ordering)
/// @throws std::invalid_argument if ordering is not valid
template <class IndexType, class IndexTag>
IndexType computeLinearIndex(Ordering ordering, const Index2D<IndexType, IndexTag>& index,
                             const Size2D<IndexType, IndexTag>& dims) {
  if (!index.isIn(dims))
    return -1;

  switch (ordering) {
    case Ordering::RowMajor:
      return index.row() * dims.cols() + index.col();
    case Ordering::ColumnMajor:
      return index.col() * dims.rows() + index.row();
  }

  throw std::invalid_argument("Ordering specified is not valid.");
}
}
}
