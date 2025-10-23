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

#include <variant>

#include <dlaf/types.h>

namespace dlaf::matrix {

enum class AllocationLayout { ColMajor, Blocks, Tiles };

enum class Ld {
  /// leading dimension is set to the minimum valid value.
  Compact,
  /// leading dimension is set to an optimal value that might introduce padding.
  Padded
};

using LdSpec = std::variant<Ld, SizeType>;

struct AllocationLayoutDefault {};
struct LdDefault {};

}
