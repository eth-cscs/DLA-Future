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

/// @file

#include <sstream>
#include <string>

namespace dlaf {
namespace common {

namespace internal {

/// This macro return a dlaf::common::internal::source_location instance using information from the line
/// where this macro is used
///
/// It uses DLAF_FUNCTION_NAME that it is set to \_\_PRETTY_FUNCTION\_\_ or \_\_func\_\_ depending on
/// availability of the former one
#define SOURCE_LOCATION()                     \
  ::dlaf::common::internal::source_location { \
    __FILE__, __LINE__, DLAF_FUNCTION_NAME    \
  }

/// Anticipation of std::source_location from C++20
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

/// Return an empty string
///
/// This is just the fundamental step of the recursive algorithm
inline std::string concat() {
  return "";
}

/// Join a list of heterogenous parameters into a string
///
/// Given a list of parameters for which a valid std::ostream& operator<<(std::ostream&, const T&)
/// exists, it returns a std::string with all parameters representations joined
template <class T, class... Ts>
std::string concat(const T& first, const Ts&... args) {
  std::ostringstream ss;
  ss << first << concat(std::forward<const Ts>(args)...);
  return ss.str();
}
}
}
