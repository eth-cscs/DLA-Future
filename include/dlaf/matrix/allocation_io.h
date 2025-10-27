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
#include <string>

#include <dlaf/matrix/allocation.h>
#include <dlaf/util_string.h>

namespace dlaf::matrix {

inline AllocationLayout allocation_layout_from(const std::string& layout) {
  std::string layout_lower = util::copy_to_lower(layout);
  if (layout_lower == "colmajor")
    return AllocationLayout::ColMajor;
  else if (layout_lower == "blocks")
    return AllocationLayout::Blocks;
  else if (layout_lower == "tiles")
    return AllocationLayout::Tiles;
  DLAF_INVALID_OPTION_VALUE("AllocationLayout", layout, "ColMajor, Blocks, Tiles (case insensitive)");
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
