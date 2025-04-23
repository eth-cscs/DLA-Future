//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <dlaf/types.h>

namespace dlaf::matrix {

enum class MatrixAllocation { ColMajor, Blocks, Tiles };

constexpr SizeType padded_ld =
    -10;  // leading dimension is set to an optimal value that might introduce padding.
constexpr SizeType compact_ld = -11;  // leading dimension is set to the minimum value

}
