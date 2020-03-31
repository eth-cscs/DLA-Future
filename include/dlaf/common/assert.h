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

#define DLAF_CHECK(category, condition, ...)                                      \
  if (!(condition)) {                                                            \
    std::cerr << "[" category "-ERROR] " << __FILE__ << ":" << __LINE__ << " : " \
              << DLAF_SOURCE_LOCATION << "\n"                                    \
              << dlaf::common::concat(#condition, ##__VA_ARGS__) << std::endl;   \
    std::terminate();                                                            \
  }

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
