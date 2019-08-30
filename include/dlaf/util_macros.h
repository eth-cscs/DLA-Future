//
// DLAF
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <type_traits>

/// @file

/// Allows to hide `std::enable_if` expressions from generated documentation.
/// If `BooleanExpr` is true it results in `Type`, otherwise it is empty (useful for exploiting SFINAE).
#ifdef DLAF_DOXYGEN
#define RETURN_TYPE_IF(Type, BooleanExpr) Type
#else
#define RETURN_TYPE_IF(Type, BooleanExpr) std::enable_if_t<BooleanExpr, Type>
#endif
