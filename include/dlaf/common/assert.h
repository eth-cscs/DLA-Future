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

#define DLAF_TEST(category, condition, ...)                                                   \
  if (!(condition)) {                                                                         \
    std::cerr << "failed " category " " #condition << "\n at " << DLAF_SOURCE_LOCATION << ":" \
              << __LINE__ << '\n'                                                             \
              << dlaf::common::concat(__VA_ARGS__) << std::endl;                              \
    std::terminate();                                                                         \
  }

#ifndef NDEBUG
#define DLAF_ASSERT(condition, ...) DLAF_TEST("assertion", condition, __VA_ARGS__)
#else
#define DLAF_ASSERT(condition, ...)
#endif

#ifdef DLAF_ENABLE_DEVASSERT
#define DLAF_DEVASSERT(condition, ...) DLAF_TEST("dev assert", condition, __VA_ARGS__)
#else
#define DLAF_DEVASSERT(condition, ...)
#endif

#define DLAF_PRECONDITION(condition, ...) DLAF_TEST("precondition", condition, __VA_ARGS__)
