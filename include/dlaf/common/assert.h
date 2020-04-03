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
    std::cerr << "[ERROR] " << location << std::endl                           \
              << dlaf::common::concat(#condition, ##__VA_ARGS__) << std::endl; \
    std::terminate();                                                          \
  }

#define DLAF_CHECK(category, condition, ...) \
  DLAF_CHECK_WITH_ORIGIN(category, (SOURCE_LOCATION()), condition, ##__VA_ARGS__)

#ifdef DLAF_ASSERT_HEAVY_ENABLE
#define DLAF_ASSERT_HEAVY(condition, ...) DLAF_CHECK("HEAVY", condition, ##__VA_ARGS__)
#else
#define DLAF_ASSERT_HEAVY(condition, ...)
#endif

#ifdef DLAF_ASSERT_MODERATE_ENABLE
#define DLAF_ASSERT_MODERATE(condition, ...) DLAF_CHECK("MODERATE", condition, ##__VA_ARGS__)
#else
#define DLAF_ASSERT_MODERATE(condition, ...)
#endif

#ifdef DLAF_ASSERT_ENABLE
#define DLAF_ASSERT_WITH_ORIGIN(location, condition, ...) \
  DLAF_CHECK_WITH_ORIGIN("", (location), condition, ##__VA_ARGS__)
#define DLAF_ASSERT(condition, ...) \
  DLAF_ASSERT_WITH_ORIGIN((SOURCE_LOCATION()), condition, ##__VA_ARGS__)
#else
#define DLAF_ASSERT_WITH_ORIGIN(location, condition, ...)
#define DLAF_ASSERT(condition, ...)
#endif
