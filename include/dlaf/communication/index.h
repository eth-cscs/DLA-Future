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

#include <dlaf/common/index2d.h>
#include <dlaf/types.h>

namespace dlaf::comm {
/// Type used for indexes in MPI API.
using IndexT_MPI = int;

/// TAG for strong-typing basic_coords.
struct TAG_MPI;

/// 2D index strong-typed for MPI.
using Index2D = common::Index2D<IndexT_MPI, TAG_MPI>;
/// 2D size strong-typed for MPI.
using Size2D = common::Size2D<IndexT_MPI, TAG_MPI>;
}
