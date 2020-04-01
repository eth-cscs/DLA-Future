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

namespace internal {

#define SOURCE_LOCATION()                     \
  ::dlaf::common::internal::source_location { \
    __FILE__, __LINE__, DLAF_FUNCTION_NAME    \
  }

struct source_location {
  const char* filename;
  const unsigned int line;
  const char* function_name;

  friend std::ostream& operator<<(std::ostream& os, const source_location& loc) {
    os << loc.filename << ":" << loc.line << " : " << loc.function_name;
    return os;
  }
};

}

inline std::string concat() {
  return "";
}

template <class T, class... Ts>
std::string concat(T&& first, Ts&&... args) {
  std::ostringstream ss;
  ss << first << " " << concat(std::forward<Ts>(args)...);
  return ss.str();
}
}
}
