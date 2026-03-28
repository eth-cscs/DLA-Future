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

#include <dlaf/memory/allocation_types.h>
#include <dlaf/tune.h>

namespace dlaf::memory {

inline AllocateOn get_allocate_on(AllocateOnType allocate_on) {
  if (std::holds_alternative<AllocateOnDefault>(allocate_on)) {
    return getTuneParameters().default_allocate_on;
  }
  return std::get<AllocateOn>(allocate_on);
}

}
