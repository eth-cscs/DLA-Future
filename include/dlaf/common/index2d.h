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
#include <type_traits>

namespace dlaf {
namespace common {

/// @brief Type specifying the leading dimension
///
/// A RowMajor ordering means that the row is the first direction to look for the next value.
/// Instead, a ColumnMajor ordering means that the column is the first direction to look for the next value
enum class Ordering { RowMajor, ColumnMajor };

namespace internal {
template <typename IndexT>
class basic_coords {
public:
  static_assert(std::is_integral<IndexT>::value && std::is_signed<IndexT>::value,
                "basic_coords just works with signed integers types");

  /// @brief Create a position with given coordinates
  basic_coords(IndexT row, IndexT col) noexcept;

  /// @brief Create a position with given coordinates
  /// @param coords where coords[0] is the row index and coords[1] is the column index
  basic_coords(const std::array<IndexT, 2>& coords) noexcept;

  /// @brief Check if it is a valid position (no upper bound check)
  /// @return true if row >= 0 and column >= 0
  bool isValid() const noexcept;

protected:
  IndexT row_;
  IndexT col_;
};
}

// forward declarations
template <typename IndexType, class Tag>
struct Size2D;

template <typename IndexT, class Tag>
class Index2D : public internal::basic_coords<IndexT> {
public:
  using IndexType = IndexT;

  /// Create an invalid position
  Index2D() noexcept;
  Index2D(IndexT row, IndexT col);
  Index2D(const std::array<IndexT, 2>& coords);

  /// @brief Check if it is a valid position inside the grid size specified by @p boundary
  /// @return true if the current index is in the range [0, @p boundary) for both row and column
  /// @pre both this and @p boundary must be valid indexes
  bool isIn(const Size2D<IndexT, Tag>& boundary) const noexcept;

  IndexT row() const noexcept {
    return internal::basic_coords<IndexT>::row_;
  }

  IndexT col() const noexcept {
    return internal::basic_coords<IndexT>::col_;
  }
};

/// @brief A tagged 2D Size
///
/// The tag is used to avoid mixing elements tagged differently
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
};

/// Compute coords of the @p index -th cell in a grid with @p ordering and sizes @p dims
/// @param dims with number of rows at @p dims[0] and number of columns at @p dims[1]
/// @param index is the linear index of the cell with specified @p ordering
template <class Index2DType, typename LinearIndexT>
Index2DType computeCoords(Ordering ordering, LinearIndexT index,
                          const std::array<typename Index2DType::IndexType, 2>& dims);

}
}

#include "index2d.ipp"
