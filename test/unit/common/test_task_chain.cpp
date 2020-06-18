//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/common/task_chain.h"

#include <gtest/gtest.h>

#include <hpx/dataflow.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/wait_all.hpp>

TEST(TaskChain, Deadlock) {
  dlaf::common::TaskChain tc{};

  // This shouldn't cause a race condition.
  int d = 0;
  auto func_1 = [&d](hpx::future<hpx::promise<void>> fp) {
    d = 1;
    fp.get().set_value();
  };
  auto fut1 = hpx::dataflow(std::move(func_1), tc.chain());

  auto func_2 = [&d](hpx::future<hpx::promise<void>> fp) {
    d = 2;
    fp.get().set_value();
  };
  auto fut2 = hpx::dataflow(std::move(func_2), tc.chain());

  hpx::wait_all(fut1, fut2);

  EXPECT_EQ(d, 2);
}
