//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include <dlaf/common/source_location.h>

namespace dlaf {
namespace internal {

/// Return an empty string.
///
/// This is just the fundamental step of the recursive algorithm.
inline std::string concat() noexcept {
  return "";
}

/// Join a list of heterogenous parameters into a string.
///
/// Given a list of parameters for which a valid std::ostream& operator<<(std::ostream&, const T&)
/// exists, it returns a std::string with all parameters representations joined.
template <class T, class... Ts>
std::string concat(const T& first, const Ts&... args) noexcept {
  std::ostringstream ss;
  ss << first << '\n' << concat(std::forward<const Ts>(args)...);
  return ss.str();
}

template <class... Ts>
inline void do_assert(bool expr, const common::internal::source_location& loc, const char* expression,
                      const Ts&... ts) noexcept {
  if (!expr) {
    std::ostringstream ss;
    ss << "[ERROR] " << loc << '\n' << expression << '\n' << concat(ts...) << "\n";
    std::cerr << ss.str();
    std::terminate();
  }
}

/// Helper function for silencing unused parameter warning.
///
/// This allows to not add std::maybe_unused to each argument of an UNIMPLEMENTED function,
/// but still keeping the name of arguments, so that in the future the signature will be
/// ready for implementation.
template <class... Args>
void silenceUnusedWarningFor(Args&&...) {}
}
}

/// Each macro has two required parameters and any number of optional parameters:
///
/// 1. @Expr        (required) : an experssion that returns a bool
/// 2. @Msg / @Var  (required) : additional message (even if empty) or variable to print
/// >2. @Msg / @Var (optional) : optional messages / variables to print
///
/// If @Expr is false, an error message is composed and `std::terminate()` is called. The error message
/// contains information about @Expr, the origin of the error and any additional messages / variables provided
/// as arguments. Any variables passed to the argument needs to be printable, i.e. support `operator<<()`.
///
/// A disabled ASSERT macro has no overhead, @Expr and any additional arguments are not evaluated.
/// Macros are enabled/disabled as follows:
///
///                       Control flag                 Default
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
/// DLAF_ASSERT(5 == 6, "");                 // ASSERT with no additional information
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
/// 3
/// 4
/// ```
///
/// Note that `my_msg()` is evaluated even if the condition is true.

#ifdef DLAF_ASSERT_ENABLE
#define DLAF_ASSERT(Expr, ...) dlaf::internal::do_assert(Expr, SOURCE_LOCATION(), #Expr, __VA_ARGS__)
#else
#define DLAF_ASSERT(Expr, ...)
#endif

#ifdef DLAF_ASSERT_MODERATE_ENABLE
#define DLAF_ASSERT_MODERATE(Expr, ...) \
  dlaf::internal::do_assert(Expr, SOURCE_LOCATION(), #Expr, __VA_ARGS__)
#else
#define DLAF_ASSERT_MODERATE(Expr, ...)
#endif

#ifdef DLAF_ASSERT_HEAVY_ENABLE
#define DLAF_ASSERT_HEAVY(Expr, ...) \
  dlaf::internal::do_assert(Expr, SOURCE_LOCATION(), #Expr, __VA_ARGS__)
#else
#define DLAF_ASSERT_HEAVY(Expr, ...)
#endif

#define DLAF_UNIMPLEMENTED(...) \
  dlaf::internal::do_assert(false, SOURCE_LOCATION(), "Not yet implemented!", __VA_ARGS__)

#define DLAF_STATIC_UNIMPLEMENTED(DummyType) \
  static_assert(sizeof(DummyType) == 0, "Not yet implemented!")

#define DLAF_STATIC_FAIL(DummyType, Msg) static_assert(sizeof(DummyType) == 0, Msg)

/// Returns a fake object of type T (specified by @p __VA_ARGS__ ) dereferencing a null pointer.
///
/// At runtime the program is exited with an error before dereferencing the null pointer.
/// This macro is useful when a return statement is needed in unreachable branches,
/// especially when T doesn't have a default constructor. (T has to be moveable.)
/// E.g. it can be used after a switch in which all the cases return
///      and all the possible options are covered.
/// Note: multiple arguments are allowed as templated types might include commas.
///       However only a type should be specified.
#define DLAF_UNREACHABLE(...)                                                     \
  dlaf::internal::do_assert(false, SOURCE_LOCATION(), "Unreachable branch hit!"), \
      std::move(*std::unique_ptr<__VA_ARGS__>())

#define DLAF_UNREACHABLE_PLAIN \
  dlaf::internal::do_assert(false, SOURCE_LOCATION(), "Unreachable branch hit!")
