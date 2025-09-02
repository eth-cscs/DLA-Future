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

#include <ostream>

#include <dlaf/matrix/allocation.h>

namespace dlaf::matrix {

inline AllocationLayout get_allocation_layout(std::string layout) {
  if (layout == "ColMajor")
    return AllocationLayout::ColMajor;
  else if (layout == "Blocks")
    return AllocationLayout::Blocks;
  else if (layout == "Tiles")
    return AllocationLayout::Tiles;
  return DLAF_UNREACHABLE(AllocationLayout);
}

inline std::ostream& operator<<(std::ostream& os, AllocationLayout layout) {
  switch (layout) {
    case AllocationLayout::ColMajor:
      os << "ColMajor";
      break;
    case AllocationLayout::Blocks:
      os << "Blocks";
      break;
    case AllocationLayout::Tiles:
      os << "Tiles";
  }
  return os;
}
}
