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

// forward declarations
namespace internal {
template <typename IndexType, class Tag>
struct basic_coords;
}

template <typename IndexType, class Tag>
struct Size2D;
template <typename IndexType, class Tag>
struct Index2D;

/// @brief Type specifying the leading dimension
///
/// A RowMajor ordering means that the row is the first direction to look for the next value.
/// Instead, a ColumnMajor ordering means that the column is the first direction to look for the next value
enum class Ordering { RowMajor, ColumnMajor };

namespace internal {
template <typename IndexType, class Tag>
class basic_coords {
  static_assert(std::is_integral<IndexType>::value && std::is_signed<IndexType>::value,
                "basic_coords just works with signed integers types");

  using index_t = IndexType;

public:
  /// Create an invalid position
  basic_coords() noexcept;

  /// @brief Create a position with given coordinates
  basic_coords(index_t row, index_t col) noexcept(false);

  /// @brief Create a position with given coordinates
  /// @param coords where coords[0] is the row index and coords[1] is the column index
  basic_coords(const std::array<index_t, 2>& coords) noexcept(false);

  /// @brief Check if it is a valid position (no upper bound check)
  /// @return true if row >= 0 and column >= 0
  bool isValid() const noexcept;

  /// @brief Return row index
  index_t row() const noexcept;

  /// @brief Return column index
  index_t col() const noexcept;

protected:
  index_t row_;
  index_t col_;
};
}

template <typename T, class Tag>
struct Size2D : public internal::basic_coords<T, Tag> {
  using internal::basic_coords<T, Tag>::basic_coords;

  template <typename, class>
  friend class Index2D;
};

template <typename T, class Tag>
struct Index2D : public internal::basic_coords<T, Tag> {
  using internal::basic_coords<T, Tag>::basic_coords;

  /// @brief Compare positions
  /// @return true if the current index is in the range [0, @p boundary) for both row and column
  /// @pre both this and @p boundary must be valid indexes
  bool isIn(const Size2D<T, Tag>& boundary) const noexcept;
};

/// Compute coords of the @p index -th cell in a grid with @p ordering and sizes @p dims
/// @param dims with number of rows at @p dims[0] and number of columns at @p dims[1]
/// @param index is the linear index of the cell with specified @p ordering
template <typename index_t, typename linear_t, class Tag>
void computeCoords(Ordering ordering, linear_t index, const std::array<index_t, 2>& dims,
                   internal::basic_coords<index_t, Tag>& coords);

}
}

#include "index2d.ipp"
