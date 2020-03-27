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
#include <sstream>

namespace dlaf {
namespace common {
namespace internal {

struct assert_helper {
  assert_helper(const char* category, const char* condition, const char* location, int line) {
    message_ << "failed " << category << " " << condition << " at " << location << ":" << line
             << std::endl;
  }

  ~assert_helper() {
    std::cerr << message_.str() << std::endl;
    std::terminate();
  }

  template <class T>
  assert_helper& operator<<(const T& part) {
    message_ << part;
    return *this;
  }

  std::stringstream message_;
};

}
}
}

#define DLAF_TEST(category, condition) \
  if (!(condition))                    \
  dlaf::common::internal::assert_helper(category, #condition, DLAF_SOURCE_LOCATION, __LINE__)

#ifndef NDEBUG
#define DLAF_ASSERT(condition) DLAF_TEST("assert", condition)
#else
#define DLAF_ASSERT(condition) DLAF_TEST("none", true)
#endif

#ifdef DLAF_ENABLE_DEVASSERT
#define DLAF_DEVASSERT(condition) DLAF_TEST("dev assert", condition)
#else
#define DLAF_DEVASSERT(condition) DLAF_TEST("none", true)
#endif

#define DLAF_PRECONDITION(condition) DLAF_TEST("precondition", condition)
