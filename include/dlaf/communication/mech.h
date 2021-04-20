//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <iostream>

namespace dlaf {
namespace comm {

enum class MPIMech { Polling, Yielding };

inline std::ostream& operator<<(std::ostream& os, const MPIMech& mech) {
  if (mech == comm::MPIMech::Polling) {
    os << "polling";
  }
  else if (mech == comm::MPIMech::Yielding) {
    os << "yielding";
  }
  return os;
}

}
}
