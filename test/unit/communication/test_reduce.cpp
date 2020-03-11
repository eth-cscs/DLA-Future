//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/common/buffer_basic.h"
#include "dlaf/communication/sync/reduce.h"

#include <gtest/gtest.h>

#include "dlaf_test/helper_communicators.h"

using namespace dlaf;
using namespace dlaf_test;
using namespace dlaf::comm;

using ReduceTest = SplittedCommunicatorsTest;

TEST_F(ReduceTest, Basic) {
  int root = 0;

  int value = 1;
  int result = 0;

  sync::reduce(root, world, MPI_SUM, make_message(common::make_buffer(&value, 1)),
               make_message(common::make_buffer(&result, 1)));

  // check that the root rank has the reduced results
  if (world.rank() == root) {
    EXPECT_EQ(NUM_MPI_RANKS * value, result);
  }

  // check that input has not been modified
  EXPECT_EQ(1, value);
}

TEST_F(ReduceTest, Multiple) {
  static_assert(NUM_MPI_RANKS >= 2, "This test requires at least two ranks");
  int root = 1;

  const std::size_t N = 3;
  int input[N] = {1, 2, 3};
  int reduced[N];

  sync::reduce(root, world, MPI_SUM, make_message(common::make_buffer(input, N)),
               make_message(common::make_buffer(reduced, N)));

  // check that the root rank has the reduced results
  if (world.rank() == root) {
    for (std::size_t index = 0; index < N; ++index)
      EXPECT_EQ(NUM_MPI_RANKS * input[index], reduced[index]);
  }

  // check that input has not been modified
  for (std::size_t index = 0; index < N; ++index)
    EXPECT_EQ(index + 1, input[index]);
}
