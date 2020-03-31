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

#include <sstream>
#include <string>

namespace dlaf {
namespace common {

std::string concat() {
  return "";
}

template <class T, class... Ts>
std::string concat(T&& first, Ts&&... args) {
  std::stringstream ss;
  ss << first << " " << concat(std::forward<Ts>(args)...);
  return ss.str();
}
}
}
