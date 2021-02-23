//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <array>
#include <cstddef>
#include <ostream>
#include <string>
#include <type_traits>

#include "dlaf/common/assert.h"
#include "dlaf/types.h"
#include "dlaf/util_math.h"

namespace dlaf {

enum class Coord { Row, Col };

// Given a Coord direction, returns its transposed
constexpr Coord transposed(const Coord rc) {
  return rc == Coord::Row ? Coord::Col : Coord::Row;
}

// Given a Coord direction, return the component that varies
constexpr Coord component(const Coord rc) {
  return rc == Coord::Row ? Coord::Col : Coord::Row;
}

constexpr auto coord2str(const Coord rc) {
  return rc == Coord::Row ? "ROW" : "COL";
}

namespace common {

/// A RowMajor ordering means that the row is the first direction to look for the next value.
/// Instead, a ColumnMajor ordering means that the column is the first direction to look for the next value.
enum class Ordering { RowMajor, ColumnMajor };

namespace internal {

/// A data structure for storing 2D coordinates (0-based).
///
/// @tparam IndexT signed integer type for row and column coordinates.
template <typename IndexT>
class basic_coords {
public:
  static_assert(std::is_integral<IndexT>::value && std::is_signed<IndexT>::value,
                "basic_coords just works with signed integers types");

  using IndexType = IndexT;

  /// Create a position with given coordinates.
  ///
  /// @param row index of the row (0-based),
  /// @param col index of the col (0-based).
  basic_coords(IndexT row, IndexT col) noexcept : row_(row), col_(col) {}

  /// Create a position with specified component (other one is set to 0)
  ///
  /// @param component specifies which component has to be set,
  /// @param value index along the @p component (0-based).
  basic_coords(Coord component, IndexT value, IndexT fixed = 0) noexcept {
    switch (component) {
      case Coord::Row:
        *this = basic_coords(value, fixed);
        break;
      case Coord::Col:
        *this = basic_coords(fixed, value);
        break;
    }
  }

  /// Create a position with given coordinates.
  ///
  /// @see basic_coords::basic_coords(IndexT row, IndexT col),
  /// @param coords where coords[0] is the row index and coords[1] is the column index.
  basic_coords(const std::array<IndexT, 2>& coords) noexcept : basic_coords(coords[0], coords[1]) {}

  /// Return a copy of the row or the col index as specified by @p rc.
  template <Coord rc>
  constexpr IndexT get() const noexcept {
    if (rc == Coord::Row)
      return row_;
    return col_;
  }

  constexpr IndexT get(const Coord rc) const noexcept {
    if (rc == Coord::Row)
      return row_;
    return col_;
  }

  /// Check if it is a valid position (no upper bound check).
  ///
  /// @return true if row >= 0 and column >= 0.
  bool isValid() const noexcept {
    return row_ >= 0 && col_ >= 0;
  }

  /// Swaps row and column index/size.
  void transpose() noexcept {
    std::swap(row_, col_);
  }

  /// Adds "(<row_>, <col_>)" to out.
  friend std::ostream& operator<<(std::ostream& out, const basic_coords& index) {
    if (std::is_same<IndexT, signed char>::value || std::is_same<IndexT, char>::value)
      return out << "(" << static_cast<int>(index.row_) << ", " << static_cast<int>(index.col_) << ")";
    return out << "(" << index.row_ << ", " << index.col_ << ")";
  }

protected:
  // NOTE: operator== and operator! are protected otherwise it would be possible to compare Index2D and
  // Size2D or same type but mixing Tag. Which is something not desired.

  /// @return true if `this` and `rhs` have the same row and column.
  bool operator==(const basic_coords& rhs) const noexcept {
    return row_ == rhs.row_ && col_ == rhs.col_;
  }

  /// @return true if `this` and `rhs` have different row or column.
  bool operator!=(const basic_coords& rhs) const noexcept {
    return !operator==(rhs);
  }

  IndexT row_;
  IndexT col_;
};

}

/// A strong-type for 2D sizes.
///
/// @tparam IndexT type for row and column coordinates,
/// @tparam Tag for strong-typing.
template <typename IndexT, class Tag>
class Size2D : public internal::basic_coords<IndexT> {
  using BaseT = internal::basic_coords<IndexT>;

public:
  using BaseT::basic_coords;

  IndexT rows() const noexcept {
    return BaseT::row_;
  }

  IndexT cols() const noexcept {
    return BaseT::col_;
  }

  /// Returns true if rows() == 0 or cols() == 0.
  ///
  /// @pre isValid().
  bool isEmpty() const noexcept {
    DLAF_ASSERT_HEAVY(BaseT::isValid(), *this);
    return rows() == 0 || cols() == 0;
  }

  /// @return true if `this` and `rhs` have the same row and column.
  bool operator==(const Size2D& rhs) const noexcept {
    return BaseT::operator==(rhs);
  }

  /// @return true if `this` and `rhs` have different row or column.
  bool operator!=(const Size2D& rhs) const noexcept {
    return BaseT::operator!=(rhs);
  }

  SizeType linear_size() const noexcept {
    return static_cast<SizeType>(BaseT::row_) * BaseT::col_;
  }
};

/// A strong-type for 2D coordinates.
///
/// @tparam IndexT type for row and column coordinates,
/// @tparam Tag for strong-typing.
template <typename IndexT, class Tag>
class Index2D : public internal::basic_coords<IndexT> {
  using BaseT = internal::basic_coords<IndexT>;

public:
  using BaseT::basic_coords;

  /// Create an invalid 2D coordinate.
  Index2D() noexcept : BaseT(-1, -1) {}

  /// Create a valid 2D coordinate,
  /// @see Index2D::Index2D(IndexT row, IndexT col).
  ///
  /// @param coords where coords[0] is the row index and coords[1] is the column index,
  /// @pre coords[0] >= 0,
  /// @pre coords[1] >= 0.
  Index2D(const std::array<IndexT, 2>& coords) noexcept : Index2D(coords[0], coords[1]) {}

  IndexT row() const noexcept {
    return BaseT::row_;
  }

  IndexT col() const noexcept {
    return BaseT::col_;
  }

  /// Check if it is a valid position inside the grid size specified by @p boundary.
  ///
  /// @param boundary size of the grid,
  /// @return true if the current index is in the range [0, @p boundary) for both row and column.
  /// @pre Index2D.isValid,
  /// @pre boundary.isValid().
  bool isIn(const Size2D<IndexT, Tag>& boundary) const noexcept {
    return this->row() < boundary.rows() && this->col() < boundary.cols() && (this->isValid()) &&
           boundary.isValid();
  }

  /// @return true if `this` and `rhs` have the same row and column.
  bool operator==(const Index2D& rhs) const noexcept {
    return BaseT::operator==(rhs);
  }

  /// @return true if `this` and `rhs` have different row or column.
  bool operator!=(const Index2D& rhs) const noexcept {
    return BaseT::operator!=(rhs);
  }
};

namespace internal {

// Traits
/// This traits has a true value if T is an Index2D or a Size2D (with any index type and any tag)
template <class T>
struct is_coord {
  constexpr static bool value = false;
};

template <class T, class Tag>
struct is_coord<Index2D<T, Tag>> {
  constexpr static bool value = true;
};

template <class T, class Tag>
struct is_coord<Size2D<T, Tag>> {
  constexpr static bool value = true;
};

}

/// Basic print utility for coordinate types
template <class Coords2DType, std::enable_if_t<internal::is_coord<Coords2DType>::value, int> = 0>
std::ostream& operator<<(std::ostream& out, const Coords2DType& index) {
  using IndexT = typename Coords2DType::IndexType;
  return out << static_cast<internal::basic_coords<IndexT>>(index);
}

/// Given a coordinate type, it returns its transpose
template <class Coords2DType, std::enable_if_t<internal::is_coord<Coords2DType>::value, int> = 0>
Coords2DType transposed(Coords2DType coords) {
  coords.transpose();
  return coords;
}

/// Compute coords of the @p index -th cell in a row-major ordered 2D grid with size @p dims.
///
/// @return an Index2D matching the Size2D (same IndexT and Tag).
/// @param dims Size2D<IndexT, Tag> representing the size of the grid,
/// @param index linear index of the cell,
/// @pre 0 <= linear_index < (dims.rows() * dims.cols()).
template <class IndexT, class Tag>
Index2D<IndexT, Tag> computeCoordsRowMajor(SizeType linear_index,
                                           const Size2D<IndexT, Tag>& dims) noexcept {
  using dlaf::util::ptrdiff_t::mul;

  // `linear_index` is wrapped with `std::to_string` because uint8_t or int8_t are interpreted as chars,
  // so the equivalent ASCII mapping is printed instead of the numeric value.
  DLAF_ASSERT_MODERATE(linear_index >= 0, std::to_string(linear_index));
  DLAF_ASSERT_MODERATE(linear_index < mul(dims.rows(), dims.cols()), std::to_string(linear_index), dims);

  SizeType leading_size = dims.cols();
  return {to_signed<IndexT>(linear_index / leading_size),
          to_signed<IndexT>(linear_index % leading_size)};
}

/// Compute coords of the @p index -th cell in a column-major ordered 2D grid with size op dims.
///
/// @return an Index2D matching the Size2D (same IndexT and Tag).
/// @param dims Size2D<IndexT, Tag> representing the size of the grid,
/// @param index linear index of the cell,
/// @pre 0 <= linear_index < (dims.rows() * dims.cols()).
template <class IndexT, class Tag>
Index2D<IndexT, Tag> computeCoordsColMajor(SizeType linear_index,
                                           const Size2D<IndexT, Tag>& dims) noexcept {
  using dlaf::util::ptrdiff_t::mul;

  DLAF_ASSERT_MODERATE(linear_index >= 0, linear_index);
  DLAF_ASSERT_MODERATE(linear_index < mul(dims.rows(), dims.cols()), std::to_string(linear_index), dims);

  SizeType leading_size = dims.rows();
  return {to_signed<IndexT>(linear_index % leading_size),
          to_signed<IndexT>(linear_index / leading_size)};
}

/// Compute coords of the @p index -th cell in a grid with @p ordering and size @p dims.
///
/// It acts as dispatcher for computeCoordsColMajor() and computeCoordsRowMajor() depending on given @p
/// ordering.
///
/// @return an Index2D matching the Size2D (same IndexT and Tag),
/// @param ordering specifies linear index layout in the grid,
/// @param dims Size2D<IndexT, Tag> representing the size of the grid,
/// @param index linear index of the cell (with specified @p ordering),
/// @pre 0 <= linear_index < (dims.rows() * dims.cols()).
template <class IndexT, class Tag>
Index2D<IndexT, Tag> computeCoords(Ordering ordering, SizeType index,
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

/// Compute linear index of an Index2D in a row-major ordered 2D grid.
///
/// The @tparam LinearIndexT cannot be deduced and it must be explicitly specified. It allows to
/// internalize the casting of the value before returning it, not leaving the burden to the user.
///
/// @tparam LinearIndexT can be any integral type signed or unsigned,
/// @pre LinearIndexT must be able to store the result,
/// @pre index.isIn(dims).
template <class LinearIndexT, class IndexT, class Tag>
LinearIndexT computeLinearIndexRowMajor(const Index2D<IndexT, Tag>& index,
                                        const Size2D<IndexT, Tag>& dims) noexcept {
  using dlaf::util::ptrdiff_t::mul;
  using dlaf::util::ptrdiff_t::sum;

  static_assert(std::is_integral<LinearIndexT>::value, "LinearIndexT must be an integral type");

  DLAF_ASSERT_MODERATE(index.isIn(dims), index, dims);

  SizeType linear_index = sum(mul(index.row(), dims.cols()), index.col());
  return integral_cast<LinearIndexT>(linear_index);
}

/// Compute linear index of an Index2D in a column-major ordered 2D grid.
///
/// The @tparam LinearIndexT cannot be deduced and it must be explicitly specified. It allows to
/// internalize the casting of the value before returning it, not leaving the burden to the user.
///
/// @tparam LinearIndexT can be any integral type signed or unsigned,
/// @pre LinearIndexT must be able to store the result,
/// @pre index.isIn(dims).
template <class LinearIndexT, class IndexT, class Tag>
LinearIndexT computeLinearIndexColMajor(const Index2D<IndexT, Tag>& index,
                                        const Size2D<IndexT, Tag>& dims) noexcept {
  using dlaf::util::ptrdiff_t::mul;
  using dlaf::util::ptrdiff_t::sum;

  static_assert(std::is_integral<LinearIndexT>::value, "LinearIndexT must be an integral type");

  DLAF_ASSERT_MODERATE(index.isIn(dims), index, dims);

  SizeType linear_index = sum(mul(index.col(), dims.rows()), index.row());
  return integral_cast<LinearIndexT>(linear_index);
}

/// Compute linear index of an Index2D.
///
/// It acts as dispatcher for computeLinearIndexColMajor() and computeLinearIndexRowMajor()
/// depending on given @p ordering.
///
/// The @tparam LinearIndexT cannot be deduced and it must be explicitly specified. It allows to
/// internalize the casting of the value before returning it, not leaving the burden to the user.
///
/// @tparam LinearIndexT can be any integral type signed or unsigned (it must be explicitly specified),
/// @pre LinearIndexT must be able to store the result,
/// @pre index.isIn(dims).
template <class LinearIndexT, class IndexT, class Tag>
LinearIndexT computeLinearIndex(Ordering ordering, const Index2D<IndexT, Tag>& index,
                                const Size2D<IndexT, Tag>& dims) noexcept {
  switch (ordering) {
    case Ordering::RowMajor:
      return computeLinearIndexRowMajor<LinearIndexT>(index, dims);
    case Ordering::ColumnMajor:
      return computeLinearIndexColMajor<LinearIndexT>(index, dims);
    default:
      return {};
  }
}

/// The following operations are defined:
///
/// Index +/- Size -> Index
/// Index - Index -> Size
/// Size +/- Size -> Size

template <class IndexT, class Tag>
Index2D<IndexT, Tag> operator+(const Index2D<IndexT, Tag>& lhs,
                               const Size2D<IndexT, Tag>& rhs) noexcept {
  return Index2D<IndexT, Tag>(lhs.row() + rhs.rows(), lhs.col() + rhs.cols());
}

template <class IndexT, class Tag>
Index2D<IndexT, Tag> operator-(const Index2D<IndexT, Tag>& lhs,
                               const Size2D<IndexT, Tag>& rhs) noexcept {
  return Index2D<IndexT, Tag>(lhs.row() - rhs.rows(), lhs.col() - rhs.cols());
}

template <class IndexT, class Tag>
Size2D<IndexT, Tag> operator-(const Index2D<IndexT, Tag>& lhs,
                              const Index2D<IndexT, Tag>& rhs) noexcept {
  return Size2D<IndexT, Tag>(lhs.row() - rhs.row(), lhs.col() - rhs.col());
}

template <class IndexT, class Tag>
Size2D<IndexT, Tag> operator-(const Size2D<IndexT, Tag>& lhs, const Size2D<IndexT, Tag>& rhs) noexcept {
  return Size2D<IndexT, Tag>(lhs.rows() - rhs.rows(), lhs.cols() - rhs.cols());
}

template <class IndexT, class Tag>
Size2D<IndexT, Tag> operator+(const Size2D<IndexT, Tag>& lhs, const Size2D<IndexT, Tag>& rhs) noexcept {
  return Size2D<IndexT, Tag>(lhs.rows() + rhs.rows(), lhs.cols() + rhs.cols());
}

}
}
