//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/common/assert.h"
#include "dlaf/types.h"
#include "dlaf/util_math.h"

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

/// Compute coords of the @p index -th cell in a grid with @p ordering and sizes @p dims
///
/// @param ordering specify linear index layout in the grid
/// @param dims Size2D matching with the given Index2D (same type and tag)
/// @param index is the linear index of the cell with specified @p ordering
template <class IndexT, class Tag, class LinearIndexT>
Index2D<IndexT, Tag> computeCoords(Ordering ordering, LinearIndexT linear_index,
                                   const Size2D<IndexT, Tag>& dims) noexcept {
  static_assert(std::is_integral<LinearIndexT>::value, "linear_index must be an integral type");

  using dlaf::util::size_t::mul;

  using UIndexT = std::make_unsigned_t<IndexT>;
  using ULinearIndexT = std::make_unsigned_t<LinearIndexT>;

  DLAF_ASSERT_HEAVY(linear_index >= 0, "The linear index cannot be negative (", linear_index, ")");
  DLAF_ASSERT_HEAVY(linear_index < mul(dims.rows(), dims.cols()), "Linear index ", linear_index,
                    " does not fit into grid ", dims);

  ULinearIndexT linear_uindex = static_cast<ULinearIndexT>(linear_index);
  UIndexT leading_size =
      static_cast<UIndexT>(ordering == Ordering::ColumnMajor ? dims.rows() : dims.cols());

  Index2D<IndexT, Tag> index;
  switch (ordering) {
    case Ordering::RowMajor:
      index = {to_signed<IndexT>(linear_uindex / leading_size),
               to_signed<IndexT>(linear_uindex % leading_size)};
      break;
    case Ordering::ColumnMajor:
      index = {to_signed<IndexT>(linear_uindex % leading_size),
               to_signed<IndexT>(linear_uindex / leading_size)};
      break;
    default:
      return {};
  }

  if (!index.isIn(dims))
    return {};

  return index;
}

/// Compute linear index of an Index2D
///
/// @return -1 if given index is outside the grid size, otherwise the linear index (w.r.t specified ordering)
template <class IndexT, class Tag>
IndexT computeLinearIndex(Ordering ordering, const Index2D<IndexT, Tag>& index,
                          const Size2D<IndexT, Tag>& dims) noexcept {
  using namespace dlaf::util::size_t;

  if (!index.isIn(dims))
    return -1;

  std::size_t linear_index;

  switch (ordering) {
    case Ordering::RowMajor:
      linear_index = sum(mul(index.row(), dims.cols()), index.col());
      break;
    case Ordering::ColumnMajor:
      linear_index = sum(mul(index.col(), dims.rows()), index.row());
      break;
    default:
      return {};
  }

  return to_signed<IndexT>(linear_index);
}
}
}
