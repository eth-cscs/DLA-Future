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

#include "dlaf/common/utils.h"

/// Each macro has two parameters:
///
/// - @Expr : an experssion that returns a bool
/// - @MsgExpr : an expression that returns a `std::string`
///
/// If @Expr is false, it prints out an error message and calls `std::terminate()`.
///
/// The error message contains information about @Expr, the origin of the error and a custom message with
/// additional information.
///
///                       Control flag                 Default
/// DLAF_CHECK            N/A                          always enabled
/// DLAF_ASSERT           DLAF_ASSERT_ENABLE           enabled
/// DLAF_ASSERT_MODERATE  DLAF_ASSERT_MODERATE_ENABLE  enabled in Debug, disabled in Release
/// DLAF_ASSERT_HEAVY     DLAF_ASSERT_HEAVY_ENABLE     enabled in Debug, disabled in Release
///
/// A disabled ASSERT macro has no overhead, @Expr and @MsgExpr are not evaluated.
///
/// Examples:
///
/// ```
/// bool is_sth_true(...) {...}
/// std::string my_msg(...) {...}
///
/// DLAF_ASSERT(5 == 6, "")                     // ASSERT with no additional information
/// DLAF_ASSERT(5 == 6, "5 is not equal to 6")  // ASSERT with additional information
/// DLAF_ASSERT(is_sth_true(), my_msg())        // ASSERT calling functions
/// ```

#define DLAF_CHECK(Expr, MsgExpr) \
  dlaf::common::internal::do_assert(Expr, SOURCE_LOCATION(), #Expr, MsgExpr)

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
