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
  basic_coords(IndexT row, IndexT col) noexcept;

  /// Create a position with given coordinates
  ///
  /// @see basic_coords::basic_coords(IndexT row, IndexT col)
  /// @param coords where coords[0] is the row index and coords[1] is the column index
  basic_coords(const std::array<IndexT, 2>& coords) noexcept;

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

  /// Adds "(<row_>, <col_>)" to out.
  friend std::ostream& operator<<(std::ostream& out, const basic_coords& index) {
    if (std::is_same<IndexT, signed char>::value) {
      return out << "(" << static_cast<int>(index.row_) << ", " << static_cast<int>(index.col_) << ")";
    }
    return out << "(" << index.row_ << ", " << index.col_ << ")";
  }

protected:
  IndexT row_;
  IndexT col_;
};

}

/// A strong-type for 2D sizes
/// @tparam IndexT type for row and column coordinates
/// @tparam Tag for strong-typing
template <typename IndexT, class Tag>
class Size2D : public internal::basic_coords<IndexT> {
  template <typename, class>
  friend class Index2D;

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
  Index2D() noexcept;

  /// Create a valid 2D coordinate
  /// @param row index of the row
  /// @param col index of the column
  /// @throw std::invalid_argument if row < 0 or col < 0
  Index2D(IndexT row, IndexT col);

  /// Create a valid 2D coordinate
  /// @see Index2D::Index2D(IndexT row, IndexT col)
  /// @param coords where coords[0] is the row index and coords[1] is the column index
  /// @throw std::invalid_argument if coords[0] < 0 or coords[1] < 0
  Index2D(const std::array<IndexT, 2>& coords);

  /// @brief Check if it is a valid position inside the grid size specified by @p boundary
  /// @param boundary size of the grid
  /// @return true if the current index is in the range [0, @p boundary) for both row and column
  /// @pre both this Index2D and @p boundary must be valid
  bool isIn(const Size2D<IndexT, Tag>& boundary) const noexcept;

  IndexT row() const noexcept {
    return internal::basic_coords<IndexT>::row_;
  }

  IndexT col() const noexcept {
    return internal::basic_coords<IndexT>::col_;
  }
};

/// Compute coords of the @p index -th cell in a grid with @p ordering and sizes @p dims
/// @param ordering specify linear index layout in the grid
/// @param dims with number of rows at @p dims[0] and number of columns at @p dims[1]
/// @param index is the linear index of the cell with specified @p ordering
template <class Index2DType, typename LinearIndexT>
Index2DType computeCoords(Ordering ordering, LinearIndexT index,
                          const std::array<typename Index2DType::IndexType, 2>& dims);

}
}

#include "index2d.tpp"
