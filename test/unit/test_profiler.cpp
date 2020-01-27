//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/profiler.h"

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

TEST(Profiler, SectionScoped) {
  using namespace dlaf::profiler;

  Manager manager = Manager::get_global_profiler();

  {
    SectionScoped _("1st", "test");
    std::this_thread::sleep_for(std::chrono::milliseconds(70));
  }

  std::thread other_thread([]() {
    SectionScoped _("2nd", "test");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  });

  {
    SectionScoped _("3rd", "test");
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
  }

  other_thread.join();
}

TEST(Profiler, WrapperTimeIt) {
  using namespace dlaf::profiler;

  Manager manager = Manager::get_global_profiler();

  bool called = false;
  auto wait = util::time_it("wrapper", "test", [&called](int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
    called = true;
  });

  EXPECT_FALSE(called);

  wait(50);

  EXPECT_TRUE(called);
}
