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

#include <exception>
#include <iostream>

#include "dlaf/common/utils.h"

#define DLAF_CHECK_WITH_ORIGIN(category, location, condition, ...)             \
  if (!(condition)) {                                                          \
    std::cerr << "[" category "-ERROR] " << location << std::endl              \
              << dlaf::common::concat(#condition, ##__VA_ARGS__) << std::endl; \
    std::terminate();                                                          \
  }

#define DLAF_CHECK(category, condition, ...) \
  DLAF_CHECK_WITH_ORIGIN(category, SOURCE_LOCATION(), condition, __VA_ARGS__)

#ifndef NDEBUG
#define DLAF_ASSERT(condition, ...) DLAF_CHECK("ASSERT", condition, ##__VA_ARGS__)
#else
#define DLAF_ASSERT(condition, ...)
#endif

#ifdef DLAF_ENABLE_AUDIT
#define DLAF_AUDIT(condition, ...) DLAF_CHECK("AUDIT", condition, ##__VA_ARGS__)
#else
#define DLAF_AUDIT(condition, ...)
#endif

#define DLAF_PRECONDITION(condition, ...) DLAF_CHECK("PRECONDITION", condition, ##__VA_ARGS__)
#define DLAF_PRECONDITION_WITH_ORIGIN(location, condition, ...) \
  DLAF_CHECK_WITH_ORIGIN("PRECONDITION", location, condition, ##__VA_ARGS__)
