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

#include <dlaf/memory/allocation.h>

namespace dlaf::memory {

inline AllocateOn get_allocate_on(const std::string& allocate_on) {
  if (allocate_on == "Construction")
    return AllocateOn::Construction;
  else if (allocate_on == "Demand")
    return AllocateOn::Demand;
  DLAF_INVALID_OPTION_VALUE("AllocateOn", allocate_on, "Construction, Demand");
  return DLAF_UNREACHABLE(AllocateOn);
}

inline std::ostream& operator<<(std::ostream& os, AllocateOn allocate_on) {
  switch (allocate_on) {
    case AllocateOn::Construction:
      os << "Construction";
      break;
    case AllocateOn::Demand:
      os << "Demand";
  }
  return os;
}
}
