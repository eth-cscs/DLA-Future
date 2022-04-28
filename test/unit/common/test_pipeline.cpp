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
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"

#include <atomic>
#include <chrono>

#include <gtest/gtest.h>
#include <pika/execution.hpp>
#include <pika/thread.hpp>

using namespace dlaf;
using namespace std::chrono_literals;

using dlaf::common::Pipeline;

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

TEST(Pipeline, Basic) {
  {
    Pipeline<int> serial(26);

    std::atomic<bool> first_access_done{false};
    std::atomic<bool> second_access_done{false};
    std::atomic<bool> third_access_done{false};

    auto checkpoint0 = serial() | ex::then([&](auto&& wrapper) {
                         EXPECT_FALSE(first_access_done);
                         EXPECT_FALSE(second_access_done);
                         EXPECT_FALSE(third_access_done);
                         first_access_done = true;
                         auto local = std::move(wrapper);
                         dlaf::internal::silenceUnusedWarningFor(local);
                       });
    auto checkpoint1 = serial() | ex::then([&](auto&& wrapper) {
                         EXPECT_TRUE(first_access_done);
                         EXPECT_FALSE(second_access_done);
                         EXPECT_FALSE(third_access_done);
                         second_access_done = true;
                         auto local = std::move(wrapper);
                         dlaf::internal::silenceUnusedWarningFor(local);
                       });
    auto checkpoint2 = serial() | ex::then([&](auto&& wrapper) {
                         EXPECT_TRUE(first_access_done);
                         EXPECT_TRUE(second_access_done);
                         EXPECT_FALSE(third_access_done);
                         third_access_done = true;
                         auto local = std::move(wrapper);
                         dlaf::internal::silenceUnusedWarningFor(local);
                       });

    tt::sync_wait(ex::when_all(std::move(checkpoint0), std::move(checkpoint1), std::move(checkpoint2)));
  }

  // The order of access does not depend on how the senders are started by when_all
  {
    Pipeline<int> serial(26);

    std::atomic<bool> first_access_done{false};
    std::atomic<bool> second_access_done{false};
    std::atomic<bool> third_access_done{false};

    auto checkpoint0 = serial() | ex::then([&](auto&& wrapper) {
                         EXPECT_FALSE(first_access_done);
                         EXPECT_FALSE(second_access_done);
                         EXPECT_FALSE(third_access_done);
                         first_access_done = true;
                         auto local = std::move(wrapper);
                         dlaf::internal::silenceUnusedWarningFor(local);
                       });
    auto checkpoint1 = serial() | ex::then([&](auto&& wrapper) {
                         EXPECT_TRUE(first_access_done);
                         EXPECT_FALSE(second_access_done);
                         EXPECT_FALSE(third_access_done);
                         second_access_done = true;
                         auto local = std::move(wrapper);
                         dlaf::internal::silenceUnusedWarningFor(local);
                       });
    auto checkpoint2 = serial() | ex::then([&](auto&& wrapper) {
                         EXPECT_TRUE(first_access_done);
                         EXPECT_TRUE(second_access_done);
                         EXPECT_FALSE(third_access_done);
                         third_access_done = true;
                         auto local = std::move(wrapper);
                         dlaf::internal::silenceUnusedWarningFor(local);
                       });

    tt::sync_wait(ex::when_all(std::move(checkpoint2), std::move(checkpoint1), std::move(checkpoint0)));
  }
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
    std::this_thread::sleep_for(wait_guard);
};

TEST(PipelineDestructor, DestructionWithDependency) {
  ex::unique_any_sender<> last_task;

  std::atomic<bool> is_exited_from_scope;
  {
    Pipeline<int> serial(26);
    last_task = dlaf::internal::transform(
                    dlaf::internal::Policy<dlaf::Backend::MC>(),
                    [&is_exited_from_scope](auto) {
                      try_waiting_guard(is_exited_from_scope);
                      EXPECT_TRUE(is_exited_from_scope);
                    },
                    serial()) |
                ex::ensure_started();
  }
  is_exited_from_scope = true;

  tt::sync_wait(std::move(last_task));
}
