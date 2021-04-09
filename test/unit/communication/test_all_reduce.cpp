//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/communication/sync/all_reduce.h"

#include <gtest/gtest.h>

#include "dlaf/common/data_descriptor.h"
#include "dlaf/communication/communicator_grid.h"

#include "dlaf_test/helper_communicators.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::comm::test;
using namespace dlaf::test;

using AllReduceTest = SplittedCommunicatorsTest;
using AllReduceInPlaceTest = SplittedCommunicatorsTest;

using TypeParam = std::complex<double>;

TEST_F(AllReduceTest, ValueOnSingleRank) {
  CommunicatorGrid alone_grid(world, 1, 1, common::Ordering::RowMajor);

  Communicator alone_world = alone_grid.rowCommunicator();

  // just the master rank has to reduce
  if (alone_world == MPI_COMM_NULL)
    return;

  constexpr int root = 0;
  constexpr TypeParam value = TypeUtilities<TypeParam>::element(13, 26);
  TypeParam result = 0;

  ASSERT_EQ(alone_world.rank(), root);

  sync::allReduce(alone_world, MPI_SUM, common::make_data(&value, 1), common::make_data(&result, 1));

  EXPECT_LE(std::abs(value - result), TypeUtilities<TypeParam>::error);
}

TEST_F(AllReduceTest, CArrayOnSingleRank) {
  CommunicatorGrid alone_grid(world, 1, 1, common::Ordering::RowMajor);

  Communicator alone_world = alone_grid.rowCommunicator();

  // just the master rank has to reduce
  if (alone_world == MPI_COMM_NULL)
    return;

  constexpr int root = 0;
  constexpr SizeType N = 3;
  constexpr TypeParam input[N] = {TypeUtilities<TypeParam>::element(0, 1),
                                  TypeUtilities<TypeParam>::element(1, 2),
                                  TypeUtilities<TypeParam>::element(2, 3)};
  TypeParam reduced[N];

  ASSERT_EQ(alone_world.rank(), root);

  sync::allReduce(alone_world, MPI_SUM, common::make_data(input, N), common::make_data(reduced, N));

  for (SizeType index = 0; index < N; ++index)
    EXPECT_LE(std::abs(input[index] - reduced[index]), TypeUtilities<TypeParam>::error);
}

TEST_F(AllReduceTest, Value) {
  constexpr TypeParam value = TypeUtilities<TypeParam>::element(13, 26);
  TypeParam result = 0;

  sync::allReduce(world, MPI_SUM, common::make_data(&value, 1), common::make_data(&result, 1));

  EXPECT_LE(std::abs(value * static_cast<TypeParam>(NUM_MPI_RANKS) - result),
            NUM_MPI_RANKS * TypeUtilities<TypeParam>::error);
}

TEST_F(AllReduceTest, CArray) {
  static_assert(NUM_MPI_RANKS >= 2, "This test requires at least two ranks");

  constexpr SizeType N = 3;
  constexpr TypeParam input[N] = {TypeUtilities<TypeParam>::element(0, 1),
                                  TypeUtilities<TypeParam>::element(1, 2),
                                  TypeUtilities<TypeParam>::element(2, 3)};
  TypeParam reduced[N];

  sync::allReduce(world, MPI_SUM, common::make_data(input, N), common::make_data(reduced, N));

  for (SizeType index = 0; index < N; ++index)
    EXPECT_LE(std::abs(input[index] * static_cast<TypeParam>(NUM_MPI_RANKS) - reduced[index]),
              NUM_MPI_RANKS * TypeUtilities<TypeParam>::error);
}

TEST_F(AllReduceTest, ContiguousToContiguous) {
  static_assert(NUM_MPI_RANKS >= 2, "This test requires at least two ranks");

  constexpr int N = 10;
  constexpr TypeParam value = TypeUtilities<TypeParam>::element(13, 26);

  TypeParam data_A[N];
  TypeParam data_B[N];

  for (auto i = 0; i < N; ++i)
    data_A[i] = value;

  auto& communicator = world;
  auto&& message_input = common::make_data(static_cast<const TypeParam*>(data_A), N);
  auto&& message_output = common::make_data(data_B, N);
  MPI_Op op = MPI_SUM;

  sync::allReduce(communicator, op, std::move(message_input), std::move(message_output));

  for (auto i = 0; i < N; ++i)
    EXPECT_LE(std::abs(value * static_cast<TypeParam>(NUM_MPI_RANKS) - data_B[i]),
              NUM_MPI_RANKS * TypeUtilities<TypeParam>::error);
}

TEST_F(AllReduceTest, StridedToContiguous) {
  static_assert(NUM_MPI_RANKS >= 2, "This test requires at least two ranks");

  // 3 blocks, 2 elements each, with a distance of 5 elements between start of each block
  // E E - - - E E - - - E E    (without padding at the end)
  constexpr SizeType nblocks = 3;
  constexpr SizeType block_size = 2;
  constexpr SizeType block_distance = 5;

  constexpr SizeType memory_footprint = (nblocks - 1) * block_distance + block_size;
  TypeParam data_strided[memory_footprint];

  constexpr int N = nblocks * block_size;
  TypeParam data_contiguous[N];

  constexpr TypeParam value = TypeUtilities<TypeParam>::element(13, 26);
  for (SizeType i_block = 0; i_block < nblocks; ++i_block)
    for (SizeType i_element = 0; i_element < block_size; ++i_element) {
      auto mem_pos = i_block * block_distance + i_element;
      data_strided[mem_pos] = value;
    }

  auto&& message_input = common::make_data(static_cast<const TypeParam*>(data_strided), nblocks,
                                           block_size, block_distance);
  auto&& message_output = common::make_data(data_contiguous, N);

  auto& communicator = world;
  MPI_Op op = MPI_SUM;
  sync::allReduce(communicator, op, std::move(message_input), std::move(message_output));

  for (auto i = 0; i < N; ++i)
    EXPECT_LE(std::abs(value * static_cast<TypeParam>(NUM_MPI_RANKS) - data_contiguous[i]),
              NUM_MPI_RANKS * TypeUtilities<TypeParam>::error);
}

TEST_F(AllReduceTest, ContiguousToStrided) {
  static_assert(NUM_MPI_RANKS >= 2, "This test requires at least two ranks");

  // 3 blocks, 2 elements each, with a distance of 5 elements between start of each block
  // E E - - - E E - - - E E    (without padding at the end)
  constexpr SizeType nblocks = 3;
  constexpr SizeType block_size = 2;
  constexpr SizeType block_distance = 5;

  constexpr SizeType memory_footprint = (nblocks - 1) * block_distance + block_size;
  TypeParam data_strided[memory_footprint];

  constexpr int N = nblocks * block_size;
  TypeParam data_contiguous[N];

  constexpr TypeParam value = TypeUtilities<TypeParam>::element(13, 26);
  for (auto i = 0; i < N; ++i)
    data_contiguous[i] = value;

  auto&& message_input = common::make_data(static_cast<const TypeParam*>(data_contiguous), N);
  auto&& message_output = common::make_data(data_strided, nblocks, block_size, block_distance);

  auto& communicator = world;
  MPI_Op op = MPI_SUM;
  sync::allReduce(communicator, op, std::move(message_input), std::move(message_output));

  for (SizeType i_block = 0; i_block < nblocks; ++i_block)
    for (SizeType i_element = 0; i_element < block_size; ++i_element) {
      auto mem_pos = i_block * block_distance + i_element;
      EXPECT_LE(std::abs(value * static_cast<TypeParam>(NUM_MPI_RANKS) - data_strided[mem_pos]),
                NUM_MPI_RANKS * TypeUtilities<TypeParam>::error);
    }
}

TEST_F(AllReduceInPlaceTest, ValueOnSingleRank) {
  CommunicatorGrid alone_grid(world, 1, 1, common::Ordering::RowMajor);

  Communicator alone_world = alone_grid.rowCommunicator();

  // just the master rank has to reduce
  if (alone_world == MPI_COMM_NULL)
    return;

  constexpr int root = 0;
  constexpr TypeParam value = TypeUtilities<TypeParam>::element(13, 26);
  TypeParam result = value;

  ASSERT_EQ(alone_world.rank(), root);

  sync::allReduceInPlace(alone_world, MPI_SUM, common::make_data(&result, 1));

  EXPECT_LE(std::abs(value - result), TypeUtilities<TypeParam>::error);
}

TEST_F(AllReduceInPlaceTest, CArrayOnSingleRank) {
  CommunicatorGrid alone_grid(world, 1, 1, common::Ordering::RowMajor);

  Communicator alone_world = alone_grid.rowCommunicator();

  // just the master rank has to reduce
  if (alone_world == MPI_COMM_NULL)
    return;

  constexpr int root = 0;
  constexpr SizeType N = 3;
  constexpr TypeParam input[N] = {TypeUtilities<TypeParam>::element(0, 1),
                                  TypeUtilities<TypeParam>::element(1, 2),
                                  TypeUtilities<TypeParam>::element(2, 3)};
  TypeParam reduced[N];
  std::copy(input, input + N, reduced);

  ASSERT_EQ(alone_world.rank(), root);

  sync::allReduceInPlace(alone_world, MPI_SUM, common::make_data(reduced, N));

  for (SizeType index = 0; index < N; ++index)
    EXPECT_LE(std::abs(input[index] - reduced[index]), TypeUtilities<TypeParam>::error);
}

TEST_F(AllReduceInPlaceTest, Value) {
  constexpr TypeParam value = TypeUtilities<TypeParam>::element(13, 26);
  TypeParam result = value;

  sync::allReduceInPlace(world, MPI_SUM, common::make_data(&result, 1));

  EXPECT_LE(std::abs(value * static_cast<TypeParam>(NUM_MPI_RANKS) - result),
            NUM_MPI_RANKS * TypeUtilities<TypeParam>::error);
}

TEST_F(AllReduceInPlaceTest, CArray) {
  static_assert(NUM_MPI_RANKS >= 2, "This test requires at least two ranks");

  constexpr SizeType N = 3;
  constexpr TypeParam input[N] = {TypeUtilities<TypeParam>::element(0, 1),
                                  TypeUtilities<TypeParam>::element(1, 2),
                                  TypeUtilities<TypeParam>::element(2, 3)};
  TypeParam reduced[N];
  std::copy(input, input + N, reduced);

  sync::allReduceInPlace(world, MPI_SUM, common::make_data(reduced, N));

  for (SizeType index = 0; index < N; ++index)
    EXPECT_LE(std::abs(input[index] * static_cast<TypeParam>(NUM_MPI_RANKS) - reduced[index]),
              NUM_MPI_RANKS * TypeUtilities<TypeParam>::error);
}

TEST_F(AllReduceInPlaceTest, WithStrided) {
  static_assert(NUM_MPI_RANKS >= 2, "This test requires at least two ranks");

  // 3 blocks, 2 elements each, with a distance of 5 elements between start of each block
  // E E - - - E E - - - E E    (without padding at the end)
  constexpr SizeType nblocks = 3;
  constexpr SizeType block_size = 2;
  constexpr SizeType block_distance = 5;

  constexpr SizeType memory_footprint = (nblocks - 1) * block_distance + block_size;
  TypeParam data_strided[memory_footprint];

  constexpr TypeParam value = TypeUtilities<TypeParam>::element(13, 26);
  for (SizeType i_block = 0; i_block < nblocks; ++i_block) {
    for (SizeType i_element = 0; i_element < block_size; ++i_element) {
      auto mem_pos = i_block * block_distance + i_element;
      data_strided[mem_pos] = value;
    }
  }

  auto message_inout = common::make_data(data_strided, nblocks, block_size, block_distance);

  sync::allReduceInPlace(world, MPI_SUM, std::move(message_inout));

  const TypeParam expected_result = value * static_cast<TypeParam>(NUM_MPI_RANKS);

  for (SizeType i_block = 0; i_block < nblocks; ++i_block) {
    for (SizeType i_element = 0; i_element < block_size; ++i_element) {
      auto mem_pos = i_block * block_distance + i_element;
      EXPECT_LE(std::abs(expected_result - data_strided[mem_pos]),
                NUM_MPI_RANKS * TypeUtilities<TypeParam>::error);
    }
  }
}
