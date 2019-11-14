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

TEST(Pool, Basic) {
  using TypeParam = int;

  Pool<TypeParam, 2> pool;
  // (ready, ready) queue: []

  Pool<TypeParam, 2>::future_t workspace3;
  {
    Pool<TypeParam, 2>::future_t workspace1, workspace2;

    workspace1 = pool.get();
    // (workspace1, ready) queue: []
    EXPECT_TRUE(workspace1.is_ready());
    EXPECT_FALSE(workspace2.is_ready());
    EXPECT_FALSE(workspace3.is_ready());

    workspace2 = pool.get();
    // (workspace1, workspace2) queue: []
    EXPECT_TRUE(workspace1.is_ready());
    EXPECT_TRUE(workspace2.is_ready());
    EXPECT_FALSE(workspace3.is_ready());

    workspace3 = pool.get();
    // (workspace1, workspace2) queue: workspace3
    EXPECT_TRUE(workspace1.is_ready());
    EXPECT_TRUE(workspace2.is_ready());
    EXPECT_FALSE(workspace3.is_ready());

    // workspace1 and workspace2, going out of scope, will free related pool resource
  }
  // (workspace3, ready) queue: []
  EXPECT_TRUE(workspace3.is_ready());

  auto workspace4 = pool.get();
  // (workspace3, workspace4) queue: []
  EXPECT_TRUE(workspace4.is_ready());

  auto workspace5 = pool.get();
  // (workspace3, workspace4) queue: [workspace5]
  EXPECT_FALSE(workspace5.is_ready());

  auto workspace6 = pool.get();
  // (workspace3, workspace4) queue: [workspace5, workspace6]
  EXPECT_FALSE(workspace6.is_ready());

  workspace4.get();
  // (workspace3, workspace5) queue: [workspace6]
  EXPECT_TRUE(workspace5.is_ready());
  EXPECT_FALSE(workspace6.is_ready());

  auto workspace7 = pool.get();
  // (workspace3, workspace5) queue: [workspace6, workspace7]
  EXPECT_TRUE(workspace3.is_ready());
  EXPECT_TRUE(workspace5.is_ready());
  EXPECT_FALSE(workspace6.is_ready());
  EXPECT_FALSE(workspace7.is_ready());

  auto workspace8 = pool.get();
  // (workspace3, workspace5) queue: [workspace6, workspace7, workspace8]
  EXPECT_TRUE(workspace3.is_ready());
  EXPECT_TRUE(workspace5.is_ready());
  EXPECT_FALSE(workspace6.is_ready());
  EXPECT_FALSE(workspace7.is_ready());
  EXPECT_FALSE(workspace8.is_ready());

  workspace5.get();
  // (workspace3, workspace6) queue: [workspace7, workspace8]
  EXPECT_TRUE(workspace3.is_ready());
  EXPECT_TRUE(workspace6.is_ready());
  EXPECT_FALSE(workspace7.is_ready());
  EXPECT_FALSE(workspace8.is_ready());
}
