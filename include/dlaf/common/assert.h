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

#define DLAF_TEST(category, condition, message)                                                 \
  if (!(condition)) {                                                                           \
    std::cerr << "failed " category " " #condition << '\n' "at " << DLAF_SOURCE_LOCATION << ":" \
              << __LINE__ << '\n'                                                               \
              << message << std::endl;                                                          \
    std::terminate();                                                                           \
  }

#ifndef NDEBUG
#define DLAF_ASSERT(condition, message) DLAF_TEST("assertion", condition, message)
#else
#define DLAF_ASSERT(condition, message)
#endif

#ifdef DLAF_ENABLE_DEVASSERT
#define DLAF_DEVASSERT(condition, message) DLAF_TEST("dev assert", condition, message)
#else
#define DLAF_DEVASSERT(condition, message)
#endif

#define DLAF_PRECONDITION(condition, message) DLAF_TEST("precondition", condition, message)
