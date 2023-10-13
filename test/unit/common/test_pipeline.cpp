//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <atomic>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <random>
#include <vector>

#include <pika/execution.hpp>
#include <pika/mutex.hpp>
#include <pika/thread.hpp>

#include <dlaf/common/pipeline.h>
#include <dlaf/sender/policy.h>
#include <dlaf/sender/transform.h>

#include <gtest/gtest.h>

using namespace dlaf;
using namespace std::chrono_literals;

using dlaf::common::Pipeline;

using Generator = std::mt19937;

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

// A nullable int is "emptied" on move construction and assignment (i.e. reset to zero) so that we can
// detect in the test if we're using a moved-from int at some point.
class nullable_size_t {
  // For testing purposes we cheat and make x mutable so that we can increment it through const
  // references. Since we externally protect access to it we will not be concurrently modifying it even
  // though the Pipeline itself gives concurrent (read-only) access.
  mutable std::size_t x = 0;

public:
  nullable_size_t() = default;
  nullable_size_t(std::size_t x) : x(x) {}

  nullable_size_t(nullable_size_t&& other) : x(std::exchange(other.x, 0)) {}
  nullable_size_t& operator=(nullable_size_t&& other) {
    x = std::exchange(other.x, 0);
    return *this;
  }

  nullable_size_t(const nullable_size_t& other) = default;
  nullable_size_t& operator=(const nullable_size_t& other) = default;

  std::size_t get() const noexcept {
    return x;
  }
  std::size_t operator++() const noexcept {
    return ++x;
  }
};

using PipelineType = Pipeline<nullable_size_t>;

TEST(Pipeline, ResetValid) {
  // The pipeline is valid after construction
  PipelineType pipeline(42);
  ASSERT_TRUE(pipeline.valid());

  // The pipeline can be reset and is invalid afterwards
  pipeline.reset();
  ASSERT_FALSE(pipeline.valid());

  // The pipeline can be reset multiple times and remains invalid
  pipeline.reset();
  ASSERT_FALSE(pipeline.valid());
}

TEST(Pipeline, Basic) {
  {
    PipelineType serial(26);

    std::atomic<bool> first_access_done{false};
    std::atomic<bool> second_access_done{false};
    std::atomic<bool> third_access_done{false};

    auto checkpoint0 = serial() | ex::then([&](auto wrapper) {
                         EXPECT_EQ(wrapper.get().get(), 26);
                         EXPECT_FALSE(first_access_done);
                         EXPECT_FALSE(second_access_done);
                         EXPECT_FALSE(third_access_done);
                         ++wrapper.get();
                         first_access_done = true;
                       });
    auto checkpoint1 = serial() | ex::then([&](auto wrapper) {
                         EXPECT_EQ(wrapper.get().get(), 27);
                         EXPECT_TRUE(first_access_done);
                         EXPECT_FALSE(second_access_done);
                         EXPECT_FALSE(third_access_done);
                         ++wrapper.get();
                         second_access_done = true;
                       });
    auto checkpoint2 = serial() | ex::then([&](auto wrapper) {
                         EXPECT_EQ(wrapper.get().get(), 28);
                         EXPECT_TRUE(first_access_done);
                         EXPECT_TRUE(second_access_done);
                         EXPECT_FALSE(third_access_done);
                         third_access_done = true;
                       });

    tt::sync_wait(ex::when_all(std::move(checkpoint0), std::move(checkpoint1), std::move(checkpoint2)));
  }

  // The order of access does not depend on how the senders are started by when_all
  {
    PipelineType serial(26);

    std::atomic<bool> first_access_done{false};
    std::atomic<bool> second_access_done{false};
    std::atomic<bool> third_access_done{false};

    auto checkpoint0 = serial() | ex::then([&](auto wrapper) {
                         EXPECT_EQ(wrapper.get().get(), 26);
                         EXPECT_FALSE(first_access_done);
                         EXPECT_FALSE(second_access_done);
                         EXPECT_FALSE(third_access_done);
                         ++wrapper.get();
                         first_access_done = true;
                       });
    auto checkpoint1 = serial() | ex::then([&](auto wrapper) {
                         EXPECT_EQ(wrapper.get().get(), 27);
                         EXPECT_TRUE(first_access_done);
                         EXPECT_FALSE(second_access_done);
                         EXPECT_FALSE(third_access_done);
                         ++wrapper.get();
                         second_access_done = true;
                       });
    auto checkpoint2 = serial() | ex::then([&](auto wrapper) {
                         EXPECT_EQ(wrapper.get().get(), 28);
                         EXPECT_TRUE(first_access_done);
                         EXPECT_TRUE(second_access_done);
                         EXPECT_FALSE(third_access_done);
                         ++wrapper.get();
                         third_access_done = true;
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

  std::atomic<bool> is_exited_from_scope{false};
  {
    PipelineType serial(26);
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

TEST(SubPipeline, Basic) {
  // A subpipeline behaves the same as a parent pipeline if the parent hasn't been used
  PipelineType pipeline(26);
  PipelineType sub_pipeline = pipeline.sub_pipeline();

  std::atomic<bool> first_access_done{false};
  std::atomic<bool> second_access_done{false};
  std::atomic<bool> third_access_done{false};

  auto checkpoint0 = sub_pipeline() | ex::then([&](auto wrapper) {
                       EXPECT_EQ(wrapper.get().get(), 26);
                       EXPECT_FALSE(first_access_done);
                       EXPECT_FALSE(second_access_done);
                       EXPECT_FALSE(third_access_done);
                       ++wrapper.get();
                       first_access_done = true;
                     });
  auto checkpoint1 = sub_pipeline() | ex::then([&](auto wrapper) {
                       EXPECT_EQ(wrapper.get().get(), 27);
                       EXPECT_TRUE(first_access_done);
                       EXPECT_FALSE(second_access_done);
                       EXPECT_FALSE(third_access_done);
                       ++wrapper.get();
                       second_access_done = true;
                     });
  auto checkpoint2 = sub_pipeline() | ex::then([&](auto wrapper) {
                       EXPECT_EQ(wrapper.get().get(), 28);
                       EXPECT_TRUE(first_access_done);
                       EXPECT_TRUE(second_access_done);
                       EXPECT_FALSE(third_access_done);
                       third_access_done = true;
                     });

  tt::sync_wait(ex::when_all(std::move(checkpoint0), std::move(checkpoint1), std::move(checkpoint2)));
}

TEST(SubPipeline, BasicParentAccess) {
  // A subpipeline will not start executing if the parent hasn't released its accesses
  PipelineType pipeline(26);

  auto first_parent_sender = pipeline();
  PipelineType sub_pipeline = pipeline.sub_pipeline();
  auto last_parent_sender = pipeline();

  std::atomic<bool> first_parent_access_done{false};
  std::atomic<bool> first_access_done{false};
  std::atomic<bool> second_access_done{false};
  std::atomic<bool> third_access_done{false};
  std::atomic<bool> last_parent_access_done{false};

  auto checkpointparent_first = std::move(first_parent_sender) | ex::then([&](auto wrapper) {
                                  EXPECT_EQ(wrapper.get().get(), 26);
                                  EXPECT_FALSE(first_parent_access_done);
                                  EXPECT_FALSE(first_access_done);
                                  EXPECT_FALSE(second_access_done);
                                  EXPECT_FALSE(third_access_done);
                                  EXPECT_FALSE(last_parent_access_done);
                                  ++wrapper.get();
                                  first_parent_access_done = true;
                                });

  auto checkpoint0 = sub_pipeline() | ex::then([&](auto wrapper) {
                       EXPECT_EQ(wrapper.get().get(), 27);
                       EXPECT_TRUE(first_parent_access_done);
                       EXPECT_FALSE(first_access_done);
                       EXPECT_FALSE(second_access_done);
                       EXPECT_FALSE(third_access_done);
                       EXPECT_FALSE(last_parent_access_done);
                       ++wrapper.get();
                       first_access_done = true;
                     });
  auto checkpoint1 = sub_pipeline() | ex::then([&](auto wrapper) {
                       EXPECT_EQ(wrapper.get().get(), 28);
                       EXPECT_TRUE(first_parent_access_done);
                       EXPECT_TRUE(first_access_done);
                       EXPECT_FALSE(second_access_done);
                       EXPECT_FALSE(third_access_done);
                       EXPECT_FALSE(last_parent_access_done);
                       ++wrapper.get();
                       second_access_done = true;
                     });
  auto checkpoint2 = sub_pipeline() | ex::then([&](auto wrapper) {
                       EXPECT_EQ(wrapper.get().get(), 29);
                       EXPECT_TRUE(first_parent_access_done);
                       EXPECT_TRUE(first_access_done);
                       EXPECT_TRUE(second_access_done);
                       EXPECT_FALSE(third_access_done);
                       EXPECT_FALSE(last_parent_access_done);
                       ++wrapper.get();
                       third_access_done = true;
                     });
  auto checkpointparent_last = std::move(last_parent_sender) | ex::then([&](auto wrapper) {
                                 EXPECT_EQ(wrapper.get().get(), 30);
                                 EXPECT_TRUE(first_parent_access_done);
                                 EXPECT_TRUE(first_access_done);
                                 EXPECT_TRUE(second_access_done);
                                 EXPECT_TRUE(third_access_done);
                                 EXPECT_FALSE(last_parent_access_done);
                                 last_parent_access_done = true;
                               });

  // The first parent access and all sub pipeline accesses should complete here. The last parent access
  // should not complete until the sub pipeline has been reset.
  auto checkpointparent_last_started = ex::ensure_started(std::move(checkpointparent_last));
  tt::sync_wait(ex::when_all(std::move(checkpoint0), std::move(checkpoint1), std::move(checkpoint2),
                             std::move(checkpointparent_first)));
  // Since the last parent access will be run as an inline continuation and the access was eagerly
  // started, it should be triggered by the reset of the sub pipeline even without a sync_wait.
  EXPECT_FALSE(last_parent_access_done);
  sub_pipeline.reset();
  EXPECT_TRUE(last_parent_access_done);
}

TEST(SubPipeline, BasicReadonlyParentAccess) {
  // A subpipeline will not start executing if the parent hasn't released its accesses
  PipelineType pipeline(26);

  auto first_parent_sender = pipeline.read();
  PipelineType sub_pipeline = pipeline.sub_pipeline();
  auto last_parent_sender = pipeline.read();

  std::atomic<bool> first_parent_access_done{false};
  std::atomic<bool> first_access_done{false};
  std::atomic<bool> second_access_done{false};
  std::atomic<bool> third_access_done{false};
  std::atomic<bool> last_parent_access_done{false};

  auto checkpointparent_first = std::move(first_parent_sender) | ex::then([&](auto wrapper) {
                                  EXPECT_EQ(wrapper.get().get(), 26);
                                  EXPECT_FALSE(first_parent_access_done);
                                  EXPECT_FALSE(first_access_done);
                                  EXPECT_FALSE(second_access_done);
                                  EXPECT_FALSE(third_access_done);
                                  EXPECT_FALSE(last_parent_access_done);
                                  ++wrapper.get();
                                  first_parent_access_done = true;
                                });

  // Read-only access can be concurrent so the order of the following tasks being executed is sensitive
  // to when they're connected and started. Since they're not eagerly started they will be released in
  // the order they are passed to when_all. Note that the order in which when_all starts senders may be
  // implementation dependent.
  //
  // In this particular test the senders are passed to when_all in the order 1, 0, 2 and the checks
  // within the tasks reflect that.
  auto checkpoint0 = sub_pipeline.read() | ex::then([&](auto wrapper) {
                       EXPECT_EQ(wrapper.get().get(), 28);
                       EXPECT_TRUE(first_parent_access_done);
                       EXPECT_FALSE(first_access_done);
                       EXPECT_TRUE(second_access_done);
                       EXPECT_FALSE(third_access_done);
                       EXPECT_FALSE(last_parent_access_done);
                       ++wrapper.get();
                       first_access_done = true;
                     });
  auto checkpoint1 = sub_pipeline.read() | ex::then([&](auto wrapper) {
                       EXPECT_EQ(wrapper.get().get(), 27);
                       EXPECT_TRUE(first_parent_access_done);
                       EXPECT_FALSE(first_access_done);
                       EXPECT_FALSE(second_access_done);
                       EXPECT_FALSE(third_access_done);
                       EXPECT_FALSE(last_parent_access_done);
                       ++wrapper.get();
                       second_access_done = true;
                     });
  auto checkpoint2 = sub_pipeline.read() | ex::then([&](auto wrapper) {
                       EXPECT_EQ(wrapper.get().get(), 29);
                       EXPECT_TRUE(first_parent_access_done);
                       EXPECT_TRUE(first_access_done);
                       EXPECT_TRUE(second_access_done);
                       EXPECT_FALSE(third_access_done);
                       EXPECT_FALSE(last_parent_access_done);
                       ++wrapper.get();
                       third_access_done = true;
                     });
  auto checkpointparent_last = std::move(last_parent_sender) | ex::then([&](auto wrapper) {
                                 EXPECT_EQ(wrapper.get().get(), 30);
                                 EXPECT_TRUE(first_parent_access_done);
                                 EXPECT_TRUE(first_access_done);
                                 EXPECT_TRUE(second_access_done);
                                 EXPECT_TRUE(third_access_done);
                                 EXPECT_FALSE(last_parent_access_done);
                                 ++wrapper.get();
                                 last_parent_access_done = true;
                               });

  // The first parent access and all sub pipeline accesses should complete here. The last parent access
  // should not complete until the sub pipeline has been reset.
  auto checkpointparent_last_started = ex::ensure_started(std::move(checkpointparent_last));
  tt::sync_wait(ex::when_all(std::move(checkpoint1), std::move(checkpoint0), std::move(checkpoint2),
                             std::move(checkpointparent_first)));
  // Since the last parent access will be run as an inline continuation and the access was eagerly
  // started, it should be triggered by the reset of the sub pipeline even without a sync_wait.
  EXPECT_FALSE(last_parent_access_done);
  sub_pipeline.reset();
  EXPECT_TRUE(last_parent_access_done);
}

TEST(SubPipeline, TaskParentAccess) {
  // A subpipeline will not start executing if the parent hasn't released its accesses
  PipelineType pipeline(26);

  auto first_parent_sender = pipeline();
  PipelineType sub_pipeline = pipeline.sub_pipeline();

  std::atomic<bool> first_parent_access_done{false};
  std::atomic<bool> first_access_done{false};
  std::atomic<bool> second_access_done{false};
  std::atomic<bool> third_access_done{false};
  std::atomic<bool> last_parent_access_done{false};

  auto checkpointparent_first = std::move(first_parent_sender) | ex::then([&](auto wrapper) {
                                  EXPECT_EQ(wrapper.get().get(), 26);
                                  EXPECT_FALSE(first_parent_access_done);
                                  EXPECT_FALSE(first_access_done);
                                  EXPECT_FALSE(second_access_done);
                                  EXPECT_FALSE(third_access_done);
                                  EXPECT_FALSE(last_parent_access_done);
                                  ++wrapper.get();
                                  first_parent_access_done = true;
                                });

  auto spawn_sub_pipeline =
      ex::just() |
      dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                [&, sub_pipeline = std::move(sub_pipeline)]() mutable {
                                  ex::start_detached(sub_pipeline() | ex::then([&](auto wrapper) {
                                                       EXPECT_EQ(wrapper.get().get(), 27);
                                                       EXPECT_TRUE(first_parent_access_done);
                                                       EXPECT_FALSE(first_access_done);
                                                       EXPECT_FALSE(second_access_done);
                                                       EXPECT_FALSE(third_access_done);
                                                       EXPECT_FALSE(last_parent_access_done);
                                                       ++wrapper.get();
                                                       first_access_done = true;
                                                     }));
                                  ex::start_detached(sub_pipeline() | ex::then([&](auto wrapper) {
                                                       EXPECT_EQ(wrapper.get().get(), 28);
                                                       EXPECT_TRUE(first_parent_access_done);
                                                       EXPECT_TRUE(first_access_done);
                                                       EXPECT_FALSE(second_access_done);
                                                       EXPECT_FALSE(third_access_done);
                                                       EXPECT_FALSE(last_parent_access_done);
                                                       ++wrapper.get();
                                                       second_access_done = true;
                                                     }));
                                  ex::start_detached(sub_pipeline() | ex::then([&](auto wrapper) {
                                                       EXPECT_EQ(wrapper.get().get(), 29);
                                                       EXPECT_TRUE(first_parent_access_done);
                                                       EXPECT_TRUE(first_access_done);
                                                       EXPECT_TRUE(second_access_done);
                                                       EXPECT_FALSE(third_access_done);
                                                       EXPECT_FALSE(last_parent_access_done);
                                                       ++wrapper.get();
                                                       third_access_done = true;
                                                     }));
                                  return std::move(sub_pipeline);
                                }) |
      ex::ensure_started();

  auto checkpointparent_last = pipeline() | ex::then([&](auto wrapper) {
                                 EXPECT_EQ(wrapper.get().get(), 30);
                                 EXPECT_TRUE(first_parent_access_done);
                                 EXPECT_TRUE(first_access_done);
                                 EXPECT_TRUE(second_access_done);
                                 EXPECT_TRUE(third_access_done);
                                 EXPECT_FALSE(last_parent_access_done);
                                 ++wrapper.get();
                                 last_parent_access_done = true;
                               });

  // None of the sub pipeline accesses should have completed at this point even if they were spawned. We
  // can start the last parent access without releasing previous accesses.
  auto checkpointparent_last_started = ex::ensure_started(std::move(checkpointparent_last));
  EXPECT_FALSE(first_access_done);
  EXPECT_FALSE(second_access_done);
  EXPECT_FALSE(third_access_done);

  // Once the first access in the parent pipeline has completed the sub pipeline accesses may complete.
  // This happens asynchronously as they were spawned in a different task.
  tt::sync_wait(std::move(checkpointparent_first));

  // The last parent access should not complete until the sub pipeline has been reset.
  EXPECT_FALSE(last_parent_access_done);
  auto sub_pipeline_from_sender = tt::sync_wait(std::move(spawn_sub_pipeline));
  EXPECT_FALSE(last_parent_access_done);
  sub_pipeline_from_sender.reset();
  tt::sync_wait(std::move(checkpointparent_last_started));
  EXPECT_TRUE(last_parent_access_done);
}

TEST(SubPipeline, TaskReadonlyParentAccess) {
  // A subpipeline will not start executing if the parent hasn't released its accesses
  PipelineType pipeline(26);

  auto first_parent_sender = pipeline.read();
  PipelineType sub_pipeline = pipeline.sub_pipeline();

  std::atomic<bool> first_parent_access_done{false};
  std::atomic<bool> first_access_done{false};
  std::atomic<bool> second_access_done{false};
  std::atomic<bool> third_access_done{false};
  std::atomic<bool> last_parent_access_done{false};

  auto checkpointparent_first = std::move(first_parent_sender) | ex::then([&](auto wrapper) {
                                  EXPECT_EQ(wrapper.get().get(), 26);
                                  EXPECT_FALSE(first_parent_access_done);
                                  EXPECT_FALSE(first_access_done);
                                  EXPECT_FALSE(second_access_done);
                                  EXPECT_FALSE(third_access_done);
                                  EXPECT_FALSE(last_parent_access_done);
                                  ++wrapper.get();
                                  first_parent_access_done = true;
                                });

  auto spawn_sub_pipeline =
      ex::just() |
      dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                [&, sub_pipeline = std::move(sub_pipeline)]() mutable {
                                  // Note also we can modify the value in the wrapper only because
                                  // nullable_int specially allows modification on const objects for
                                  // testing purposes.
                                  ex::start_detached(sub_pipeline.read() | ex::then([&](auto wrapper) {
                                                       EXPECT_GE(wrapper.get().get(), 27);
                                                       EXPECT_LE(wrapper.get().get(), 27);
                                                       EXPECT_TRUE(first_parent_access_done);
                                                       EXPECT_FALSE(last_parent_access_done);
                                                       ++wrapper.get();
                                                       first_access_done = true;
                                                     }));
                                  ex::start_detached(sub_pipeline.read() | ex::then([&](auto wrapper) {
                                                       EXPECT_GE(wrapper.get().get(), 27);
                                                       EXPECT_LE(wrapper.get().get(), 29);
                                                       EXPECT_TRUE(first_parent_access_done);
                                                       EXPECT_FALSE(last_parent_access_done);
                                                       ++wrapper.get();
                                                       second_access_done = true;
                                                     }));
                                  ex::start_detached(sub_pipeline.read() | ex::then([&](auto wrapper) {
                                                       EXPECT_GE(wrapper.get().get(), 27);
                                                       EXPECT_LE(wrapper.get().get(), 29);
                                                       EXPECT_TRUE(first_parent_access_done);
                                                       EXPECT_FALSE(last_parent_access_done);
                                                       ++wrapper.get();
                                                       third_access_done = true;
                                                     }));
                                  return std::move(sub_pipeline);
                                }) |
      ex::ensure_started();

  auto checkpointparent_last = pipeline.read() | ex::then([&](auto wrapper) {
                                 EXPECT_EQ(wrapper.get().get(), 30);
                                 EXPECT_TRUE(first_parent_access_done);
                                 EXPECT_TRUE(first_access_done);
                                 EXPECT_TRUE(second_access_done);
                                 EXPECT_TRUE(third_access_done);
                                 EXPECT_FALSE(last_parent_access_done);
                                 ++wrapper.get();
                                 last_parent_access_done = true;
                               });

  // None of the sub pipeline accesses should have completed at this point even if they were spawned. We
  // can start the last parent access without releasing previous accesses.
  auto checkpointparent_last_started = ex::ensure_started(std::move(checkpointparent_last));
  EXPECT_FALSE(first_access_done);
  EXPECT_FALSE(second_access_done);
  EXPECT_FALSE(third_access_done);

  // Once the first access in the parent pipeline has completed the sub pipeline accesses may complete.
  // This happens asynchronously as they were spawned in a different task.
  tt::sync_wait(std::move(checkpointparent_first));

  // The last parent access should not complete until the sub pipeline has been reset.
  EXPECT_FALSE(last_parent_access_done);
  auto sub_pipeline_from_sender = tt::sync_wait(std::move(spawn_sub_pipeline));
  EXPECT_FALSE(last_parent_access_done);
  sub_pipeline_from_sender.reset();
  tt::sync_wait(std::move(checkpointparent_last_started));
  EXPECT_TRUE(last_parent_access_done);
}

enum class SubPipelineAccessType { inline_access, new_task };

struct PipelineTestConfig {
  std::size_t min_readwrite_accesses;
  std::size_t max_readwrite_accesses;
  std::size_t min_readonly_accesses;
  std::size_t max_readonly_accesses;
  SubPipelineAccessType access_type;
};

const std::vector<PipelineTestConfig> pipeline_tests({
    // No nested access
    {0, 0, 0, 0, SubPipelineAccessType::inline_access},

    // Read-write access
    {0, 1, 0, 0, SubPipelineAccessType::inline_access},
    {0, 5, 0, 0, SubPipelineAccessType::inline_access},
    {5, 10, 0, 0, SubPipelineAccessType::inline_access},

    // Read-only access
    {0, 0, 0, 1, SubPipelineAccessType::inline_access},
    {0, 0, 0, 5, SubPipelineAccessType::inline_access},
    {0, 0, 5, 10, SubPipelineAccessType::inline_access},

    // Mixed
    {0, 1, 0, 1, SubPipelineAccessType::inline_access},
    {0, 1, 1, 5, SubPipelineAccessType::inline_access},
    {1, 5, 0, 1, SubPipelineAccessType::inline_access},
    {5, 10, 5, 10, SubPipelineAccessType::inline_access},

    // Same with subpipelines in new tasks
    {0, 0, 0, 0, SubPipelineAccessType::new_task},

    // Read-write access
    {0, 1, 0, 0, SubPipelineAccessType::new_task},
    {0, 5, 0, 0, SubPipelineAccessType::new_task},
    {5, 10, 0, 0, SubPipelineAccessType::new_task},

    // Read-only access
    {0, 0, 0, 1, SubPipelineAccessType::new_task},
    {0, 0, 0, 5, SubPipelineAccessType::new_task},
    {0, 0, 5, 10, SubPipelineAccessType::new_task},

    // Mixed
    {0, 1, 0, 1, SubPipelineAccessType::new_task},
    {0, 1, 1, 5, SubPipelineAccessType::new_task},
    {1, 5, 0, 1, SubPipelineAccessType::new_task},
    {5, 10, 5, 10, SubPipelineAccessType::new_task},
});

struct PipelineTestState {
  std::atomic<std::size_t> spawn_count{0};
  std::atomic<std::size_t> access_count{0};
  std::vector<ex::unique_any_sender<>> accesses{};
  pika::mutex accesses_mutex;
  pika::mutex count_mutex;
};

void spawn_work_readwrite(PipelineType& pipeline, PipelineTestState& state, Generator& gen,
                          const PipelineTestConfig& test) {
  std::uniform_int_distribution<std::size_t> dist(test.min_readwrite_accesses,
                                                  test.max_readwrite_accesses);

  for (std::size_t i = 0; i < dist(gen); ++i) {
    ++state.spawn_count;
    auto s = pipeline() |
             dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                       [&state, spawn_count = state.spawn_count.load()](auto& count) {
                                         ++state.access_count;
                                         ++count;

                                         EXPECT_EQ(state.access_count.load(), spawn_count);
                                         EXPECT_EQ(count.get(), spawn_count);
                                       }) |
             ex::ensure_started();

    std::lock_guard l(state.accesses_mutex);
    state.accesses.emplace_back(std::move(s));
  }
}

void spawn_work_readonly(PipelineType& pipeline, PipelineTestState& state, Generator& gen,
                         const PipelineTestConfig& test) {
  std::uniform_int_distribution<std::size_t> dist(test.min_readonly_accesses,
                                                  test.max_readonly_accesses);

  const std::size_t num_readonly_accesses = dist(gen);
  const std::size_t min_expected_count = state.spawn_count + 1;
  const std::size_t max_expected_count = state.spawn_count + 1 + num_readonly_accesses;
  for (std::size_t i = 0; i < num_readonly_accesses; ++i) {
    ++state.spawn_count;
    auto s = pipeline.read() |
             dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                       [&state, min_expected_count, max_expected_count,
                                        spawn_count = state.spawn_count.load()](auto& count) {
                                         const auto current_access_count = ++state.access_count;

                                         const auto current_count = [&] {
                                           std::lock_guard l(state.count_mutex);
                                           return ++count;
                                         }();

                                         EXPECT_GE(current_access_count, min_expected_count);
                                         EXPECT_LT(current_access_count, max_expected_count);
                                         EXPECT_GE(current_count, min_expected_count);
                                         EXPECT_LT(current_count, max_expected_count);
                                       }) |
             ex::ensure_started();

    std::lock_guard l(state.accesses_mutex);
    state.accesses.emplace_back(std::move(s));
  }
}

void spawn_work(PipelineType& pipeline, PipelineTestState& state, Generator& gen,
                const PipelineTestConfig& test) {
  // Note that any of the blocks below may spawn zero accesses depending the test configuration
  spawn_work_readwrite(pipeline, state, gen, test);
  spawn_work_readonly(pipeline, state, gen, test);
  spawn_work_readwrite(pipeline, state, gen, test);
}

void test_recurse_sub_pipeline(PipelineType& pipeline, PipelineTestState& state, Generator& gen,
                               const PipelineTestConfig& test, std::size_t remaining_depth) {
  if (remaining_depth == 0) {
    return;
  }

  spawn_work(pipeline, state, gen, test);

  auto recurse_f = [&state, &test, gen, remaining_depth,
                    sub_pipeline = pipeline.sub_pipeline()]() mutable {
    test_recurse_sub_pipeline(sub_pipeline, state, gen, test, remaining_depth - 1);
  };

  switch (test.access_type) {
    case SubPipelineAccessType::inline_access:
      recurse_f();
      break;
    case SubPipelineAccessType::new_task:
      ex::start_detached(ex::just() |
                         dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                                   std::move(recurse_f)));
      break;
  }

  // If we are using subpipelines in new tasks we can't keep track of the correct counts anymore after
  // spawning the task with the subpipeline
  if (test.access_type == SubPipelineAccessType::inline_access) {
    spawn_work(pipeline, state, gen, test);
  }
}

TEST(SubPipeline, RandomAccess) {
  constexpr std::size_t depth = 5;
  constexpr std::size_t num_seeds = 10;

  for (const auto& test : pipeline_tests) {
    for (std::size_t seed = 0; seed < num_seeds; ++seed) {
      std::mt19937 gen(seed);

      PipelineTestState state{};
      PipelineType pipeline(0);

      // Make the first access to the parent pipeline, but do not eagerly start it
      ++state.spawn_count;
      auto first_access = pipeline() | ex::then([&state](auto wrapper) {
                            ++state.access_count;
                            ++wrapper.get();

                            EXPECT_EQ(state.access_count.load(), 1);
                            EXPECT_EQ(wrapper.get().get(), 1);
                          });

      test_recurse_sub_pipeline(pipeline, state, gen, test, depth);

      // After the work has been created the spawn count should be at least one and equal to the number
      // of senders stored in accesses. No accesses should be done yet at this point since they depend
      // on the first access completing.
      ASSERT_GE(state.spawn_count, 1);
      ASSERT_GE(state.spawn_count, state.accesses.size());
      EXPECT_EQ(state.access_count.load(), 0);

      // After the first access is done there can be one or more accesses done
      tt::sync_wait(std::move(first_access));
      EXPECT_GE(state.access_count.load(), 1);

      // After all accesses have completed we expect there to have been more than one access. We can
      // only check the following if we are spawning work inline.
      if (test.access_type == SubPipelineAccessType::inline_access) {
        tt::sync_wait(ex::when_all_vector(std::move(state.accesses)));

        if (test.min_readwrite_accesses > 0 || test.min_readonly_accesses) {
          EXPECT_GT(state.access_count.load(), 1);
        }

        // We should have accessed the pipeline value the same number of times as spawns
        EXPECT_EQ(state.access_count.load(), state.spawn_count);
      }

      // The count held by the pipeline should have been incremented spawn_count times by now
      auto final_count = tt::sync_wait(pipeline()).get().get();
      EXPECT_EQ(state.spawn_count, final_count);
    }
  }
}
