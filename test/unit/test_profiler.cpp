//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "profiling/profiler.h"

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

using namespace dlaf::profiling;

TEST(Profiler, ProfileScope) {
  profiler::instance();

  {
    profile_scope _("1st", "test");
    std::this_thread::sleep_for(std::chrono::milliseconds(70));
  }

  std::thread other_thread([]() {
    profile_scope _("2nd", "test");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  });

  {
    profile_scope _("3rd", "test");
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
  }

  other_thread.join();
}

TEST(Profiler, WrapperTimeIt) {
  profiler::instance();

  bool called = false;
  auto wait = util::time_it("wrapper", "test", [&called](int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
    called = true;
  });

  EXPECT_FALSE(called);

  wait(50);

  EXPECT_TRUE(called);
}
