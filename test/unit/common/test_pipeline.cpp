//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/common/pipeline.h"

#include <gtest/gtest.h>

#include <hpx/include/util.hpp>
#include <hpx/local/future.hpp>

using namespace dlaf;
using dlaf::common::Pipeline;

TEST(Pipeline, Basic) {
  Pipeline<int> serial(26);

  auto checkpoint0 = serial();
  auto checkpoint1 =
      checkpoint0.then(hpx::launch::sync,
                       hpx::util::unwrapping([](auto&& wrapper) { return std::move(wrapper); }));

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

TEST(Pipeline, DestructionNoDependency) {
  Pipeline<int> serial(13);
}

TEST(Pipeline, DestructionWithDependency) {
  Pipeline<int> serial(26);
  auto checkpoint = serial();
}
