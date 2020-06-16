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

TEST(Pipeline, Basic) {
  dlaf::common::TaskChain tc{};

  int d = 0;
  auto func_1 = [&d](hpx::future<hpx::promise<void>> fp) {
    fp.get().set_value();
    d = 1;
  };
  auto fut1 = hpx::dataflow(std::move(func_1), tc.chain());

  auto func_2 = [&d](hpx::future<hpx::promise<void>> fp) {
    fp.get().set_value();
    d = 2;
  };
  auto fut2 = hpx::dataflow(std::move(func_2), tc.chain());

  hpx::wait_all(fut1, fut2);

  EXPECT_EQ(d, 2);
}
