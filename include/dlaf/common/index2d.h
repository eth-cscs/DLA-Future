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

/// @file

#include <array>
#include <cassert>
#include <ostream>
#include <type_traits>

#include "dlaf/common/assert.h"
#include "dlaf/types.h"
#include "dlaf/util_math.h"

namespace dlaf {

enum class Coord { Row, Col };

namespace common {

/// A RowMajor ordering means that the row is the first direction to look for the next value.
/// Instead, a ColumnMajor ordering means that the column is the first direction to look for the next value
enum class Ordering { RowMajor, ColumnMajor };

namespace internal {

/// A data structure for storing 2D coordinates (0-based)
/// @tparam IndexT signed integer type for row and column coordinates
template <typename IndexT>
class basic_coords {
public:
  static_assert(std::is_integral<IndexT>::value && std::is_signed<IndexT>::value,
                "basic_coords just works with signed integers types");

  /// Create a position with given coordinates
  ///
  /// @param row index of the row (0-based)
  /// @param col index of the col (0-based)
  basic_coords(IndexT row, IndexT col) noexcept : row_(row), col_(col) {}

  /// Create a position with given coordinates
  ///
  /// @see basic_coords::basic_coords(IndexT row, IndexT col)
  /// @param coords where coords[0] is the row index and coords[1] is the column index
  basic_coords(const std::array<IndexT, 2>& coords) noexcept : basic_coords(coords[0], coords[1]) {}

  /// Compare two indices.
  ///
  /// @return true if row and column index of *this and rhs are equal.
  bool operator==(const basic_coords& rhs) const noexcept {
    return row_ == rhs.row_ && col_ == rhs.col_;
  }

  /// Compare two indices.
  ///
  /// @return true if any of row and column index of *this and rhs are different.
  bool operator!=(const basic_coords& rhs) const noexcept {
    return !operator==(rhs);
  }

  /// Return a copy of the row or the col index as specified by @p rc
  template <Coord rc>
  IndexT get() const noexcept {
    if (rc == Coord::Row)
      return row_;
    return col_;
  }

  /// Check if it is a valid position (no upper bound check)
  ///
  /// @return true if row >= 0 and column >= 0
  bool isValid() const noexcept {
    return row_ >= 0 && col_ >= 0;
  }

  /// Swaps row and column index/size.
  void transpose() noexcept {
    std::swap(row_, col_);
  }

  /// Given a coordinate, returns its transposed with the same type
  template <class Coords2DType>
  friend Coords2DType transposed(const Coords2DType& coords);

  /// Adds "(<row_>, <col_>)" to out.
  friend std::ostream& operator<<(std::ostream& out, const basic_coords& index) {
    if (std::is_same<IndexT, signed char>::value || std::is_same<IndexT, char>::value) {
      return out << "(" << static_cast<int>(index.row_) << ", " << static_cast<int>(index.col_) << ")";
    }
    return out << "(" << index.row_ << ", " << index.col_ << ")";
  }

protected:
  IndexT row_;
  IndexT col_;
};

template <class Coords2DType>
Coords2DType transposed(const Coords2DType& coords) {
  return {coords.col_, coords.row_};
}
}

/// A strong-type for 2D sizes
/// @tparam IndexT type for row and column coordinates
/// @tparam Tag for strong-typing
template <typename IndexT, class Tag>
class Size2D : public internal::basic_coords<IndexT> {
public:
  using internal::basic_coords<IndexT>::basic_coords;

  IndexT rows() const noexcept {
    return internal::basic_coords<IndexT>::row_;
  }

  IndexT cols() const noexcept {
    return internal::basic_coords<IndexT>::col_;
  }

  /// @brief Returns true if rows() == 0 or cols() == 0
  /// @pre isValid() == true
  bool isEmpty() const noexcept {
    assert(internal::basic_coords<IndexT>::isValid());
    return rows() == 0 || cols() == 0;
  }
};

/// A strong-type for 2D coordinates
/// @tparam IndexT type for row and column coordinates
/// @tparam Tag for strong-typing
template <typename IndexT, class Tag>
class Index2D : public internal::basic_coords<IndexT> {
public:
  using IndexType = IndexT;

  /// Create an invalid 2D coordinate
  Index2D() noexcept : internal::basic_coords<IndexT>(-1, -1) {}

  /// Create a valid 2D coordinate
  /// @param row index of the row
  /// @param col index of the column
  /// @throw std::invalid_argument if row < 0 or col < 0
  Index2D(IndexT row, IndexT col) : internal::basic_coords<IndexT>(row, col) {
    if (!internal::basic_coords<IndexT>::isValid())
      throw std::invalid_argument("indices are not valid (negative).");
  }

  /// Create a valid 2D coordinate
  /// @see Index2D::Index2D(IndexT row, IndexT col)
  /// @param coords where coords[0] is the row index and coords[1] is the column index
  /// @throw std::invalid_argument if coords[0] < 0 or coords[1] < 0
  Index2D(const std::array<IndexT, 2>& coords) : Index2D(coords[0], coords[1]) {}

  /// @brief Check if it is a valid position inside the grid size specified by @p boundary
  /// @param boundary size of the grid
  /// @return true if the current index is in the range [0, @p boundary) for both row and column
  /// @pre both this Index2D and @p boundary must be valid
  bool isIn(const Size2D<IndexT, Tag>& boundary) const noexcept {
    return this->row() < boundary.rows() && this->col() < boundary.cols() && (this->isValid()) &&
           boundary.isValid();
  }

  IndexT row() const noexcept {
    return internal::basic_coords<IndexT>::row_;
  }

  IndexT col() const noexcept {
    return internal::basic_coords<IndexT>::col_;
  }
};

template <class IndexT, class Tag, class LinearIndexT>
Index2D<IndexT, Tag> computeCoordsRowMajor(LinearIndexT index,
                                           const Size2D<IndexT, Tag>& dims) noexcept {
  return {static_cast<IndexT>(index / dims.cols()), static_cast<IndexT>(index % dims.cols())};
}

template <class IndexT, class Tag, class LinearIndexT>
Index2D<IndexT, Tag> computeCoordsColMajor(LinearIndexT index,
                                           const Size2D<IndexT, Tag>& dims) noexcept {
  return {static_cast<IndexT>(index % dims.rows()), static_cast<IndexT>(index / dims.rows())};
}

/// Compute coords of the @p index -th cell in a grid with @p ordering and sizes @p dims
///
/// Return an Index2D matching the Size2D (same IndexT and Tag)
/// @param ordering specify linear index layout in the grid
/// @param dims Size2D<IndexT, Tag>
/// @param index is the linear index of the cell with specified @p ordering
template <class IndexT, class Tag, class LinearIndexT>
Index2D<IndexT, Tag> computeCoords(Ordering ordering, LinearIndexT index,
                                   const Size2D<IndexT, Tag>& dims) noexcept {
  switch (ordering) {
    case Ordering::RowMajor:
      return computeCoordsRowMajor(index, dims);
    case Ordering::ColumnMajor:
      return computeCoordsColMajor(index, dims);
    default:
      return {};
  }
}

template <class IndexT, class Tag>
IndexT computeLinearIndexRowMajor(const Index2D<IndexT, Tag>& index,
                                  const Size2D<IndexT, Tag>& dims) noexcept {
  DLAF_ASSERT_MODERATE(index.isIn(dims), "Index ", index, " is not in the grid ", dims);

  using dlaf::util::size_t::mul;
  using dlaf::util::size_t::sum;

  std::size_t linear_index = sum(mul(index.row(), dims.cols()), index.col());

  return to_signed<IndexT>(linear_index);
}

template <class IndexT, class Tag>
IndexT computeLinearIndexColMajor(const Index2D<IndexT, Tag>& index,
                                  const Size2D<IndexT, Tag>& dims) noexcept {
  DLAF_ASSERT_MODERATE(index.isIn(dims), "Index ", index, " is not in the grid ", dims);

  using dlaf::util::size_t::mul;
  using dlaf::util::size_t::sum;

  std::size_t linear_index = sum(mul(index.col(), dims.rows()), index.row());

  return to_signed<IndexT>(linear_index);
}

/// Compute linear index of an Index2D
///
/// @pre index.isIn(dims)
template <class IndexT, class Tag>
IndexT computeLinearIndex(Ordering ordering, const Index2D<IndexT, Tag>& index,
                          const Size2D<IndexT, Tag>& dims) noexcept {
  switch (ordering) {
    case Ordering::RowMajor:
      return computeLinearIndexRowMajor(index, dims);
    case Ordering::ColumnMajor:
      return computeLinearIndexColMajor(index, dims);
    default:
      return {};
  }
}

}
}
