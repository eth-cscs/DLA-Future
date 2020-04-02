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
  DLAF_CHECK_WITH_ORIGIN(category, (SOURCE_LOCATION()), condition, ##__VA_ARGS__)

#ifdef DLAF_ENABLE_ASSERT_HIGH
#define DLAF_ASSERT_HIGH(condition, ...) DLAF_CHECK("HIGH", condition, ##__VA_ARGS__)
#else
#define DLAF_ASSERT_HIGH(condition, ...)
#endif

#ifdef DLAF_ENABLE_ASSERT_MED
#define DLAF_ASSERT_MED(condition, ...) DLAF_CHECK("MEDIUM", condition, ##__VA_ARGS__)
#else
#define DLAF_ASSERT_MED(condition, ...)
#endif

#ifdef DLAF_ENABLE_ASSERT_LOW
#define DLAF_ASSERT_LOW_WITH_ORIGIN(location, condition, ...) DLAF_CHECK_WITH_ORIGIN("LOW", (location), condition, ##__VA_ARGS__)
#define DLAF_ASSERT_LOW(condition, ...) DLAF_ASSERT_LOW_WITH_ORIGIN((SOURCE_LOCATION()), condition, ##__VA_ARGS__)
#else
#define DLAF_ASSERT_LOW_WITH_ORIGIN(location, condition, ...)
#define DLAF_ASSERT_LOW(condition, ...)
#endif
