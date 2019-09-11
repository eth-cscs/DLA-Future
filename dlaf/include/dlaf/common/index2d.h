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

#include <array>

namespace dlaf {
namespace common {

/// 2D coordinates
struct Index2D {
  /// Create an invalid position
  Index2D() noexcept;

  /// @brief Create a position with given coordinates
  Index2D(int row, int col) noexcept;

  /// @brief Create a position with given coordinates
  ///
  /// Where coords[0] is the row index and coords[1] is the column index
  Index2D(const std::array<int, 2> & coords) noexcept;

  /// @brief Check if it is a valid position (no upper bound check)
  ///
  /// @return true if row >= 0 and column >= 0
  bool isValid() const noexcept;

  /// @brief Compare positions
  ///
  /// @return true if the current index is in the range [0, boundary) for both row and column
  ///
  /// @pre both this and @p boundary must be valid indexes
  bool operator <(const Index2D & boundary) const noexcept;

  /// @brief Return row index
  int row() const noexcept;

  /// @brief Return column index
  int col() const noexcept;

  protected:
  int row_;
  int col_;
};

}
}
