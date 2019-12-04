//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <gtest/gtest.h>

#include "dlaf/common/pool.h"

using dlaf::common::Pool;

TEST(Pool, UsageExample) {
  using TypeParam = std::unique_ptr<int>;

  Pool<TypeParam> pool(2);
  // (ready, ready) queue: []

  Pool<TypeParam>::future_t workspace2;
  {
    Pool<TypeParam>::future_t workspace0, workspace1;

    workspace0 = pool.get();
    // (workspace0, ready) queue: []
    EXPECT_TRUE(workspace0.is_ready());
    EXPECT_FALSE(workspace1.is_ready());
    EXPECT_FALSE(workspace2.is_ready());

    workspace1 = pool.get();
    // (workspace0, workspace1) queue: []
    EXPECT_TRUE(workspace0.is_ready());
    EXPECT_TRUE(workspace1.is_ready());
    EXPECT_FALSE(workspace2.is_ready());

    workspace2 = pool.get();
    // (workspace0, workspace1) queue: workspace2
    EXPECT_TRUE(workspace0.is_ready());
    EXPECT_TRUE(workspace1.is_ready());
    EXPECT_FALSE(workspace2.is_ready());

    // workspace0 and workspace1, going out of scope, will free related pool resource
  }
  // (workspace2, ready) queue: []
  EXPECT_TRUE(workspace2.is_ready());

  auto workspace3 = pool.get();
  // (workspace2, workspace3) queue: []
  EXPECT_TRUE(workspace2.is_ready());
  EXPECT_TRUE(workspace3.is_ready());

  auto workspace4 = pool.get();
  // (workspace2, workspace3) queue: [workspace4]
  EXPECT_TRUE(workspace2.is_ready());
  EXPECT_TRUE(workspace3.is_ready());
  EXPECT_FALSE(workspace4.is_ready());

  auto workspace5 = pool.get();
  // (workspace2, workspace3) queue: [workspace4, workspace5]
  EXPECT_TRUE(workspace2.is_ready());
  EXPECT_TRUE(workspace3.is_ready());
  EXPECT_FALSE(workspace4.is_ready());
  EXPECT_FALSE(workspace5.is_ready());

  {
    auto data_wrapper = workspace3.get();
    data_wrapper() = std::make_unique<int>(13);
  }
  // (workspace2, workspace4) queue: [workspace5]
  EXPECT_TRUE(workspace2.is_ready());
  EXPECT_TRUE(workspace4.is_ready());
  EXPECT_FALSE(workspace5.is_ready());

  auto workspace6 = pool.get();
  // (workspace2, workspace4) queue: [workspace5, workspace6]
  EXPECT_TRUE(workspace2.is_ready());
  EXPECT_TRUE(workspace4.is_ready());
  EXPECT_FALSE(workspace5.is_ready());
  EXPECT_FALSE(workspace6.is_ready());

  auto workspace7 = pool.get();
  // (workspace2, workspace4) queue: [workspace5, workspace6, workspace7]
  EXPECT_TRUE(workspace2.is_ready());
  EXPECT_TRUE(workspace4.is_ready());
  EXPECT_FALSE(workspace5.is_ready());
  EXPECT_FALSE(workspace6.is_ready());
  EXPECT_FALSE(workspace7.is_ready());

  workspace4.get();
  // (workspace2, workspace5) queue: [workspace6, workspace7]
  EXPECT_TRUE(workspace2.is_ready());
  EXPECT_TRUE(workspace5.is_ready());
  EXPECT_FALSE(workspace6.is_ready());
  EXPECT_FALSE(workspace7.is_ready());

  {
    auto data_wrapper = workspace5.get();
    EXPECT_EQ(13, *data_wrapper());
  }
  // (workspace2, workspace6) queue: [workspace7]
  EXPECT_TRUE(workspace2.is_ready());
  EXPECT_TRUE(workspace6.is_ready());
  EXPECT_FALSE(workspace7.is_ready());

  auto workspace8 = pool.get();
  // (workspace2, workspace6) queue: [workspace7, workspace8]
  EXPECT_TRUE(workspace2.is_ready());
  EXPECT_TRUE(workspace6.is_ready());
  EXPECT_FALSE(workspace7.is_ready());
  EXPECT_FALSE(workspace8.is_ready());

  auto workspace9 = pool.get();
  // (workspace2, workspace6) queue: [workspace7, workspace8, workspace9]
  EXPECT_TRUE(workspace2.is_ready());
  EXPECT_TRUE(workspace6.is_ready());
  EXPECT_FALSE(workspace7.is_ready());
  EXPECT_FALSE(workspace8.is_ready());
  EXPECT_FALSE(workspace9.is_ready());

  {
    auto data_wrapper = workspace2.get();
    data_wrapper() = std::make_unique<int>(26);
  }
  // (workspace7, workspace6) queue: [workspace8, workspace9]
  EXPECT_TRUE(workspace7.is_ready());
  EXPECT_TRUE(workspace6.is_ready());
  EXPECT_FALSE(workspace8.is_ready());
  EXPECT_FALSE(workspace9.is_ready());

  auto workspaceA = pool.get();
  // (workspace7, workspace6) queue: [workspace8, workspace9, workspaceA]
  EXPECT_TRUE(workspace7.is_ready());
  EXPECT_TRUE(workspace6.is_ready());
  EXPECT_FALSE(workspace8.is_ready());
  EXPECT_FALSE(workspace9.is_ready());
  EXPECT_FALSE(workspaceA.is_ready());

  auto workspaceB = pool.get();
  // (workspace7, workspace6) queue: [workspace8, workspace9, workspaceA, workspaceB]
  EXPECT_TRUE(workspace7.is_ready());
  EXPECT_TRUE(workspace6.is_ready());
  EXPECT_FALSE(workspace8.is_ready());
  EXPECT_FALSE(workspace9.is_ready());
  EXPECT_FALSE(workspaceA.is_ready());
  EXPECT_FALSE(workspaceB.is_ready());

  {
    auto data_wrapper = workspace7.get();
    EXPECT_EQ(26, *data_wrapper());
  }
  // (workspace8, workspace6) queue: [workspace9, workspaceA, workspaceB]
  EXPECT_TRUE(workspace8.is_ready());
  EXPECT_TRUE(workspace6.is_ready());
  EXPECT_FALSE(workspace9.is_ready());
  EXPECT_FALSE(workspaceA.is_ready());
  EXPECT_FALSE(workspaceB.is_ready());

  // going out of scope, everything will be released
  // (workspace8, workspace6) queue: [workspace9, workspaceA, workspaceB]
}
