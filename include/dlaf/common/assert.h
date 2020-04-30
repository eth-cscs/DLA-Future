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

#include <exception>
#include <iostream>

#include "dlaf/common/utils.h"

/// If @p condition is false, it prints out an error message and calls std::terminate()
///
/// The error message contains information about the @p condition, the @p origin of the error as specified
/// by the parameter and an optional custom message composed by concatenating all extra parameters.
/// Message composition is lazily evaluated and it will not add any overhead if the condition is true.
///
/// **This check cannot be disabled**
#define DLAF_CHECK_WITH_ORIGIN(category, origin, condition, ...)                    \
  if (!(condition)) {                                                               \
    std::cerr << "[ERROR] " << origin << std::endl                                  \
              << dlaf::common::concat(#condition, ' ', ##__VA_ARGS__) << std::endl; \
    std::terminate();                                                               \
  }

/// This macro is a shortcut for #DLAF_CHECK_WITH_ORIGIN, it sets automatically the origin to the line
/// from where this macro is used
#define DLAF_CHECK(category, condition, ...) \
  DLAF_CHECK_WITH_ORIGIN(category, (SOURCE_LOCATION()), condition, ##__VA_ARGS__)

#ifdef DLAF_ASSERT_HEAVY_ENABLE
/// **THIS MACRO MUST BE USED WHEN THE CHECK IS NEEDED FOR DEBUGGING PURPOSES THAT HAS
/// HIGH IMPACT ON PERFORMANCES.**
///
/// If the condition is false, it will print an error report and call std::terminate()
///
/// The error report can be extended with a custom message composed concatenating additional parameters
/// given to the macro.
///
/// If the switch **DLAF_ASSERT_HEAVY_ENABLE** is not defined, this check will not
/// be performed and it will not add any overhead, nor for the condition evaluation, nor for the message
#define DLAF_ASSERT_HEAVY(...) DLAF_CHECK("HEAVY", __VA_ARGS__)
#else
#define DLAF_ASSERT_HEAVY(...)
#endif

#ifdef DLAF_ASSERT_MODERATE_ENABLE
/// **THIS MACRO MUST BE USED WHEN THE CHECK IS NEEDED FOR DEBUGGING PURPOSES THAT HAVE
/// MODERATE IMPACT ON PERFORMANCES.**
///
/// If the condition is false, it will print an error report and call std::terminate()
///
/// The error report can be extended with a custom message composed concatenating additional parameters
/// given to the macro.
///
/// If the switch **DLAF_ASSERT_MODERATE_ENABLE** is not defined, this check
/// will not be performed and it will not add any overhead, nor for the condition evaluation, nor for the message
#define DLAF_ASSERT_MODERATE(...) DLAF_CHECK("MODERATE", __VA_ARGS__)
#else
#define DLAF_ASSERT_MODERATE(...)
#endif

#ifdef DLAF_ASSERT_ENABLE
/// **THIS MACRO MUST BE USED WHEN THE CHECK IS
/// NEEDED TO ENSURE A CONDITION THAT HAVE VERY LOW IMPACT ON PERFORMANCES.**
///
/// If the condition is false, it will print an error report and call std::terminate()
///
/// The error report will refer to the given origin and can be extended with a custom message composed
/// concatenating additional parameters given to the macro.
///
/// If the switch **DLAF_ASSERT_ENABLE** is not defined, this check will not be performed and it will not
/// add any overhead, nor for the condition evaluation, nor for the message
#define DLAF_ASSERT_WITH_ORIGIN(origin, ...) DLAF_CHECK_WITH_ORIGIN("", origin, __VA_ARGS__)

/// **THIS MACRO MUST BE USED WHEN THE CHECK IS NEEDED TO ENSURE A CONDITION THAT HAVE
/// VERY LOW IMPACT ON PERFORMANCES.**
///
/// If the condition is false, it will print an error report and call std::terminate()
///
/// The error report can be extended with a custom message composed concatenating additional parameters
/// given to the macro.
///
/// If the switch **DLAF_ASSERT_ENABLE** is not defined, this check will not
/// be performed and it will not add any overhead, nor for the condition evaluation, nor for the message
#define DLAF_ASSERT(...) DLAF_ASSERT_WITH_ORIGIN((SOURCE_LOCATION()), __VA_ARGS__)
#else
#define DLAF_ASSERT_WITH_ORIGIN(origin, ...)

#define DLAF_ASSERT(...)
#endif
