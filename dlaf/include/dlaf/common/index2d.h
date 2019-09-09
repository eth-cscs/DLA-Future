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
class Index2D {
  public:
  Index2D() noexcept;
  Index2D(const std::array<int, 2> & coords) noexcept;
  Index2D(int row, int col) noexcept;

  int row() const noexcept;
  int col() const noexcept;

  protected:
  int row_;
  int col_;
};

}
}
