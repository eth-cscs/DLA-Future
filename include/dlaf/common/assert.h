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

#include <cstdio>

#define DLAF_TEST(condition, message, ...)                           \
  if (!(condition)) {                                                \
    std::fprintf(stderr, "%s:%d\n", DLAF_SOURCE_LOCATION, __LINE__); \
    std::fprintf(stderr, "%s\n", #condition);                        \
    std::fprintf(stderr, message, ##__VA_ARGS__);                    \
    std::fprintf(stderr, "%s\n", "");                                \
    std::terminate();                                                \
  }

#ifndef NDEBUG
#define DLAF_ASSERT(condition, message, ...) DLAF_TEST(condition, message, ##__VA_ARGS__)
#else
#define DLAF_ASSERT(condition, message, ...)
#endif

#ifdef DLAF_ENABLE_DEVASSERT
#define DLAF_DEVASSERT(condition, message, ...) DLAF_TEST(condition, message, ##__VA_ARGS__)
#else
#define DLAF_DEVASSERT(condition, message, ...)
#endif

#define DLAF_PRECONDITION(condition, message, ...) DLAF_TEST(condition, message, ##__VA_ARGS__)
