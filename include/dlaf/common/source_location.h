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

#include <iostream>
#include <sstream>
#include <string>

namespace dlaf {
namespace common {

namespace internal {

/// This macro return a dlaf::common::internal::source_location instance using information from the line
/// where this macro is used.
///
/// It uses DLAF_FUNCTION_NAME that it is set to \_\_PRETTY_FUNCTION\_\_ or \_\_func\_\_ depending on
/// availability of the former one.
#define SOURCE_LOCATION()                     \
  ::dlaf::common::internal::source_location { \
    __FILE__, __LINE__, DLAF_FUNCTION_NAME    \
  }

/// Anticipation of std::source_location from C++20.
///
/// see https://en.cppreference.com/w/cpp/utility/source_location
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

}
}
