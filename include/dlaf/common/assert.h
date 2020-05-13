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

#include <cstring>
#include <exception>
#include <iostream>

#include "dlaf/common/utils.h"

// TODO: reword documentation
/// TODO: repase: It sets automatically the origin to the line from where this macro is used.

// TODO: Expr must return bool
// TODO: MsgExpr must return std::string

/// If @p condition is false, it prints out an error message and calls std::terminate()
///
/// The error message contains information about the @p condition, the @p origin of the error as specified
/// by the parameter and an optional custom message composed by concatenating all extra parameters.
/// Message composition is lazily evaluated and it will not add any overhead if the condition is true.
///
/// No newline is appended to the given message, which cannot be empty.
///
/// **This check cannot be disabled**
#define DLAF_CHECK(Expr, MsgExpr) do_assert(Expr, SOURCE_LOCATION(), #Expr, MsgExpr)

// TODO: reword documentation
/// **THIS MACRO MUST BE USED WHEN THE CHECK IS NEEDED FOR DEBUGGING PURPOSES THAT HAVE
/// MODERATE IMPACT ON PERFORMANCES.**
///
/// Parameters:
/// 1     condition
/// 2-*   (optional) comma separated part(s) composing the custom message in case of failure
///
/// If the condition is false, it will print an error report and call std::terminate()
///
/// The error report can be extended with a custom message composed concatenating additional parameters
/// given to the macro.
///
/// If the switch **DLAF_ASSERT_MODERATE_ENABLE** is not defined, this check
/// will not be performed and it will not add any overhead, nor for the condition evaluation, nor for the message

#ifdef DLAF_ASSERT_HEAVY_ENABLE
#define DLAF_ASSERT_HEAVY(Expr, MsgExpr) DLAF_CHECK(Expr, MsgExpr)
#else
#define DLAF_ASSERT_HEAVY(Expr, MsgExpr)
#endif

#ifdef DLAF_ASSERT_MODERATE_ENABLE
#define DLAF_ASSERT_MODERATE(Expr, MsgExpr) DLAF_CHECK(Expr, MsgExpr)
#else
#define DLAF_ASSERT_MODERATE(Expr, MsgExpr)
#endif

#ifdef DLAF_ASSERT_ENABLE
#define DLAF_ASSERT(Expr, MsgExpr) DLAF_CHECK(Expr, MsgExpr)
#else
#define DLAF_ASSERT(Expr, MsgExpr)
#endif
