//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/common/pipeline.h"

#include <atomic>
#include <chrono>

#include <gtest/gtest.h>
#include <pika/future.hpp>
#include <pika/thread.hpp>
#include <pika/unwrap.hpp>

using namespace dlaf;
using namespace std::chrono_literals;

using dlaf::common::Pipeline;

TEST(Pipeline, Basic) {
  Pipeline<int> serial(26);

  auto checkpoint0 = serial();
  auto checkpoint1 =
      checkpoint0.then(pika::launch::sync,
                       pika::unwrapping([](auto&& wrapper) { return std::move(wrapper); }));

  auto guard0 = serial();
  auto guard1 = serial();

  EXPECT_TRUE(checkpoint1.is_ready());
  EXPECT_FALSE(guard0.is_ready());
  EXPECT_FALSE(guard1.is_ready());

  checkpoint1.get();

  EXPECT_TRUE(guard0.is_ready());
  EXPECT_FALSE(guard1.is_ready());

  guard0.get();

  EXPECT_TRUE(guard1.is_ready());

  guard1.get();
}

// PipelineDestructor
//
// These tests checks that the Pipeline does not block on destruction is performed correctly.
//
// Note 1:
// In each task there is the last_task future that must depend on the launched task. This is needed
// in order to being able to wait for it before the test ends, otherwise it may end after the test is
// already finished (and in case of failure it may not be presented correctly)
//
// Note 2:
// wait_guard is the time to wait in the launched task for assuring that Pipeline d'tor has been called
// after going out-of-scope. This duration must be kept as low as possible in order to not waste time
// during tests, but at the same time it must be enough to let the "main" to arrive to the end of the
// scope.

// wait for guard to become true
auto try_waiting_guard = [](auto& guard) {
  const auto wait_guard = 20ms;

  for (int i = 0; i < 100 && !guard; ++i)
    pika::this_thread::sleep_for(wait_guard);
};

TEST(PipelineDestructor, DestructionWithDependency) {
  pika::future<void> last_task;

  std::atomic<bool> is_exited_from_scope;
  {
    Pipeline<int> serial(26);
    last_task = serial().then(pika::launch::async, [&is_exited_from_scope](auto) {
      try_waiting_guard(is_exited_from_scope);
      EXPECT_TRUE(is_exited_from_scope);
    });
  }
  is_exited_from_scope = true;

  last_task.get();
}
