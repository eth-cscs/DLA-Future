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

#include <algorithm>

#include "dlaf/common/utils.h"

namespace dlaf {
namespace internal {

inline void do_assert(bool expr, const common::internal::source_location& loc, const char* expression,
                      std::string const& msg) {
  if (!expr) {
    std::cerr << "[ERROR] " << loc << '\n' << expression << '\n' << msg << std::endl;
    std::terminate();
  }
}

template <class T>
std::string concat(const std::string& a_str, const std::string& delim, const T& a) {
  std::stringstream ss;
  ss << a_str << delim << a;
  return ss.str();
}

// The overload takes precedence over the template definition, it handles cases such as
// `concat("\"sth\"", ' ', "sth")` where the first argument is the stringified version of the last.
inline std::string concat(const std::string& a_str, const std::string& delim, const char a[]) {
  // if `a_str` without the extra `"` is the same as `a`, return `a_str`
  if (std::equal(a_str.begin() + 1, a_str.end() - 1, a)) {
    return a_str;
  }

  std::stringstream ss;
  ss << a_str << delim << a;
  return ss.str();
}

}
}

// Utility macro to concatenate as follows: "#ARG : ARG\nSTH"
#define DLAF_CONCAT(ARG, STH) dlaf::internal::concat(dlaf::internal::concat(#ARG, " : ", ARG), "\n", STH)

// Utility macro to call `do_assert()` and pass "STH"
#define DLAF_CHECK_INVOKE(Expr, STH) dlaf::internal::do_assert(Expr, SOURCE_LOCATION(), #Expr, STH)

// Utility to invoke the correct macro overload for up to 7 parameters
#define DLAF_GET_MACRO(_1, _2, _3, _4, _5, _6, _7, NAME, ...) NAME

// A set of macros to compose the error message
// clang-format off
#define DLAF_FE_1(ARG     ) DLAF_CONCAT(ARG, "")
#define DLAF_FE_2(ARG, ...) DLAF_CONCAT(ARG, DLAF_FE_1(__VA_ARGS__))
#define DLAF_FE_3(ARG, ...) DLAF_CONCAT(ARG, DLAF_FE_2(__VA_ARGS__))
#define DLAF_FE_4(ARG, ...) DLAF_CONCAT(ARG, DLAF_FE_3(__VA_ARGS__))
#define DLAF_FE_5(ARG, ...) DLAF_CONCAT(ARG, DLAF_FE_4(__VA_ARGS__))
#define DLAF_FE_6(ARG, ...) DLAF_CONCAT(ARG, DLAF_FE_5(__VA_ARGS__))

#define DLAF_CHECK_1(Expr     ) DLAF_CHECK_INVOKE(Expr, "")
#define DLAF_CHECK_2(Expr, ...) DLAF_CHECK_INVOKE(Expr, DLAF_FE_1(__VA_ARGS__))
#define DLAF_CHECK_3(Expr, ...) DLAF_CHECK_INVOKE(Expr, DLAF_FE_2(__VA_ARGS__))
#define DLAF_CHECK_4(Expr, ...) DLAF_CHECK_INVOKE(Expr, DLAF_FE_3(__VA_ARGS__))
#define DLAF_CHECK_5(Expr, ...) DLAF_CHECK_INVOKE(Expr, DLAF_FE_4(__VA_ARGS__))
#define DLAF_CHECK_6(Expr, ...) DLAF_CHECK_INVOKE(Expr, DLAF_FE_5(__VA_ARGS__))
#define DLAF_CHECK_7(Expr, ...) DLAF_CHECK_INVOKE(Expr, DLAF_FE_6(__VA_ARGS__))
// clang-format on

/// Each macro has at least one and up to seven parameter:
///
/// 1.   @Expr    (required) : an experssion that returns a bool
/// 2-7. @ARG1-6  (optional) : additional arguments to print
///
/// If @Expr is false, an error message is composed and `std::terminate()` is called. The error message
/// contains information about @Expr, the origin of the error, and up to 6 optional arguments. If any of
/// @ARG1-6 is a raw string (e.g. "my msg"), the message is printed directly, otherwise the type of the
/// argument needs to be printable, i.e. support `operator<<()`.
///
/// A disabled ASSERT macro has no overhead, @Expr and any additional arguments are not evaluated.
/// Macros are enabled/disabled as follows:
///
///                       Control flag                 Default
/// DLAF_CHECK            N/A                          always enabled
/// DLAF_ASSERT           DLAF_ASSERT_ENABLE           enabled
/// DLAF_ASSERT_MODERATE  DLAF_ASSERT_MODERATE_ENABLE  enabled in Debug, disabled in Release
/// DLAF_ASSERT_HEAVY     DLAF_ASSERT_HEAVY_ENABLE     enabled in Debug, disabled in Release
///
/// Examples:
///
/// ```
/// bool is_sth_true(...) {...}
/// std::string my_msg(...) {...}
/// int a = 3;
/// int b = 4;
///
/// DLAF_ASSERT(5 == 6);                     // ASSERT with no additional information
/// DLAF_ASSERT(a == b, "my msg", a, b);     // * ASSERT with additional information and argument
/// DLAF_ASSERT(is_sth_true(), my_msg());    // ASSERT calling functions
/// ```
///
/// The output from * is as follows:
//
/// ```
/// [ERROR] <file_name>:<line_number> : <function>
/// a == b
/// my msg
/// a : 3
/// b : 4
/// ```

// This uses a macro trick to invoke the correct overload depending on the number of arguments
#define DLAF_CHECK(...)                                                                             \
  DLAF_GET_MACRO(__VA_ARGS__, DLAF_CHECK_7, DLAF_CHECK_6, DLAF_CHECK_5, DLAF_CHECK_4, DLAF_CHECK_3, \
                 DLAF_CHECK_2, DLAF_CHECK_1)                                                        \
  (__VA_ARGS__)

#ifdef DLAF_ASSERT_HEAVY_ENABLE
#define DLAF_ASSERT_HEAVY(...) DLAF_CHECK(__VA_ARGS__)
#else
#define DLAF_ASSERT_HEAVY(...)
#endif

#ifdef DLAF_ASSERT_MODERATE_ENABLE
#define DLAF_ASSERT_MODERATE(...) DLAF_CHECK(__VA_ARGS__)
#else
#define DLAF_ASSERT_MODERATE(...)
#endif

#ifdef DLAF_ASSERT_ENABLE
#define DLAF_ASSERT(...) DLAF_CHECK(__VA_ARGS__)
#else
#define DLAF_ASSERT(...)
#endif
