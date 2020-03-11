//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/common/data_descriptor.h"
#include "dlaf/communication/sync/reduce.h"

#include <gtest/gtest.h>

#include "dlaf_test/helper_communicators.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf_test;
using namespace dlaf::comm;

using ReduceTest = SplittedCommunicatorsTest;

TEST_F(ReduceTest, Basic) {
  int root = 0;

  int value = 1;
  int result = 0;

  sync::reduce(root, world, MPI_SUM, common::make_data(&value, 1), common::make_data(&result, 1));

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

  sync::reduce(root, world, MPI_SUM, common::make_data(input, N), common::make_data(reduced, N));

  // check that the root rank has the reduced results
  if (world.rank() == root) {
    for (std::size_t index = 0; index < N; ++index)
      EXPECT_EQ(NUM_MPI_RANKS * input[index], reduced[index]);
  }

  // check that input has not been modified
  for (std::size_t index = 0; index < N; ++index)
    EXPECT_EQ(index + 1, input[index]);
}

TEST_F(ReduceTest, ContiguousToContiguous) {
  using TypeParam = int;

  const int N = 10;
  const TypeParam value = 13;

  TypeParam data_A[N];
  TypeParam data_B[N];

  for (auto i = 0; i < N; ++i)
    data_A[i] = value;

  const int root = 0;
  auto& communicator = world;
  auto&& message_input = common::make_data(data_A, N);
  auto&& message_output = common::make_data(data_B, N);
  MPI_Op op = MPI_SUM;

  sync::reduce(root, communicator, op, std::move(message_input), std::move(message_output));

  if (root == world.rank()) {
    for (auto i = 0; i < N; ++i)
      EXPECT_EQ(value * NUM_MPI_RANKS, data_B[i]);
  }
}

TEST_F(ReduceTest, ConstContiguousToContiguous) {
  using TypeParam = int;

  const int N = 10;
  const TypeParam value = 13;

  TypeParam data_A[N];
  const TypeParam* const_data_A = data_A;
  TypeParam data_B[N];

  for (auto i = 0; i < N; ++i)
    data_A[i] = value;

  const int root = 0;
  auto& communicator = world;
  auto&& message_input = common::make_data(const_data_A, N);
  auto&& message_output = common::make_data(data_B, N);
  MPI_Op op = MPI_SUM;

  sync::reduce(root, communicator, op, std::move(message_input), std::move(message_output));

  if (root == world.rank()) {
    for (auto i = 0; i < N; ++i)
      EXPECT_EQ(value * NUM_MPI_RANKS, data_B[i]);
  }
}

TEST_F(ReduceTest, StridedToContiguous) {
  using TypeParam = int;

  const TypeParam value = 13;

  // 3 blocks, 2 elements each, with a distance of 5 elements between start of each block
  // E E - - - E E - - - E E    (without padding at the end)
  const std::size_t nblocks = 3;
  const std::size_t block_size = 2;
  const std::size_t block_distance = 5;

  const std::size_t memory_footprint = (nblocks - 1) * block_distance + block_size;
  TypeParam data_strided[memory_footprint];

  const int N = nblocks * block_size;
  TypeParam data_contiguous[N];

  for (std::size_t i_block = 0; i_block < nblocks; ++i_block)
    for (std::size_t i_element = 0; i_element < block_size; ++i_element) {
      auto mem_pos = i_block * block_distance + i_element;
      data_strided[mem_pos] = dlaf_test::TypeUtilities<TypeParam>::element(value, 0);
    }

  const int root = 0;
  auto& communicator = world;
  auto&& message_input = common::make_data(data_strided, nblocks, block_size, block_distance);
  auto&& message_output = common::make_data(data_contiguous, N);
  MPI_Op op = MPI_SUM;

  sync::reduce(root, communicator, op, std::move(message_input), std::move(message_output));

  if (root == world.rank()) {
    for (auto i = 0; i < N; ++i)
      EXPECT_EQ(value * NUM_MPI_RANKS, data_contiguous[i]);
  }
}

TEST_F(ReduceTest, ConstStridedToContiguous) {
  using TypeParam = int;

  const TypeParam value = 13;

  // 3 blocks, 2 elements each, with a distance of 5 elements between start of each block
  // E E - - - E E - - - E E    (without padding at the end)
  const std::size_t nblocks = 3;
  const std::size_t block_size = 2;
  const std::size_t block_distance = 5;

  const std::size_t memory_footprint = (nblocks - 1) * block_distance + block_size;
  TypeParam data_strided[memory_footprint];
  const TypeParam* const_data_strided = data_strided;

  const int N = nblocks * block_size;
  TypeParam data_contiguous[N];

  for (std::size_t i_block = 0; i_block < nblocks; ++i_block)
    for (std::size_t i_element = 0; i_element < block_size; ++i_element) {
      auto mem_pos = i_block * block_distance + i_element;
      data_strided[mem_pos] = dlaf_test::TypeUtilities<TypeParam>::element(value, 0);
    }

  const int root = 0;
  auto& communicator = world;
  auto&& message_input = common::make_data(const_data_strided, nblocks, block_size, block_distance);
  auto&& message_output = common::make_data(data_contiguous, N);
  MPI_Op op = MPI_SUM;

  sync::reduce(root, communicator, op, std::move(message_input), std::move(message_output));

  if (root == world.rank()) {
    for (auto i = 0; i < N; ++i)
      EXPECT_EQ(value * NUM_MPI_RANKS, data_contiguous[i]);
  }
}

TEST_F(ReduceTest, ContiguousToStrided) {
  using TypeParam = int;

  const TypeParam value = 13;

  // 3 blocks, 2 elements each, with a distance of 5 elements between start of each block
  // E E - - - E E - - - E E    (without padding at the end)
  const std::size_t nblocks = 3;
  const std::size_t block_size = 2;
  const std::size_t block_distance = 5;

  const std::size_t memory_footprint = (nblocks - 1) * block_distance + block_size;
  TypeParam data_strided[memory_footprint];

  const int N = nblocks * block_size;
  TypeParam data_contiguous[N];

  for (auto i = 0; i < N; ++i)
    data_contiguous[i] = dlaf_test::TypeUtilities<TypeParam>::element(value, 0);

  const int root = 0;
  auto& communicator = world;
  auto&& message_input = common::make_data(data_contiguous, N);
  auto&& message_output = common::make_data(data_strided, nblocks, block_size, block_distance);
  MPI_Op op = MPI_SUM;

  sync::reduce(root, communicator, op, std::move(message_input), std::move(message_output));
  if (world.rank() == root) {
    for (std::size_t i_block = 0; i_block < nblocks; ++i_block)
      for (std::size_t i_element = 0; i_element < block_size; ++i_element) {
        auto mem_pos = i_block * block_distance + i_element;
        EXPECT_EQ(value * NUM_MPI_RANKS, data_strided[mem_pos]);
      }
  }
}

TEST_F(ReduceTest, ConstContiguousToStrided) {
  using TypeParam = int;

  const TypeParam value = 13;

  // 3 blocks, 2 elements each, with a distance of 5 elements between start of each block
  // E E - - - E E - - - E E    (without padding at the end)
  const std::size_t nblocks = 3;
  const std::size_t block_size = 2;
  const std::size_t block_distance = 5;

  const std::size_t memory_footprint = (nblocks - 1) * block_distance + block_size;
  TypeParam data_strided[memory_footprint];

  const int N = nblocks * block_size;
  TypeParam data_contiguous[N];
  const TypeParam* const_data_contiguous = data_contiguous;

  for (auto i = 0; i < N; ++i)
    data_contiguous[i] = dlaf_test::TypeUtilities<TypeParam>::element(value, 0);

  const int root = 0;
  auto& communicator = world;
  auto&& message_input = common::make_data(const_data_contiguous, N);
  auto&& message_output = common::make_data(data_strided, nblocks, block_size, block_distance);
  MPI_Op op = MPI_SUM;

  sync::reduce(root, communicator, op, std::move(message_input), std::move(message_output));
  if (world.rank() == root) {
    for (std::size_t i_block = 0; i_block < nblocks; ++i_block)
      for (std::size_t i_element = 0; i_element < block_size; ++i_element) {
        auto mem_pos = i_block * block_distance + i_element;
        EXPECT_EQ(value * NUM_MPI_RANKS, data_strided[mem_pos]);
      }
  }
}
