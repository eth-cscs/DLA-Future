//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/communication/sync/reduce.h"

#include <gtest/gtest.h>

#include "dlaf/common/data_descriptor.h"
#include "dlaf/communication/communicator_grid.h"

#include "dlaf_test/helper_communicators.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf_test;
using namespace dlaf::comm;

using ReduceTest = SplittedCommunicatorsTest;

template <class T, class = std::enable_if_t<std::is_arithmetic<T>::value>>
void DLAF_EXPECT_NEAR(const T a, const T b, const T abs_err) {
  EXPECT_NEAR(a, b, abs_err);
}

template <class T>
void DLAF_EXPECT_NEAR(const std::complex<T>& a, const std::complex<T>& b, const T abs_err) {
  EXPECT_NEAR(std::abs(a - b), 0, abs_err);
}

using TypeParam = std::complex<double>;

TEST_F(ReduceTest, ValueOnSingleRank) {
  CommunicatorGrid alone_grid(world, 1, 1, common::Ordering::RowMajor);

  Communicator alone_world = alone_grid.rowCommunicator();

  // just the master rank has to reduce
  if (alone_world == MPI_COMM_NULL)
    return;

  constexpr int root = 0;
  constexpr TypeParam value = TypeUtilities<TypeParam>::element(13, 26);
  TypeParam result = 0;

  ASSERT_EQ(alone_world.rank(), root);

  sync::reduce(root, alone_world, MPI_SUM, common::make_data(&value, 1), common::make_data(&result, 1));

  DLAF_EXPECT_NEAR(value, result, TypeUtilities<TypeParam>::error);
}

TEST_F(ReduceTest, CArrayOnSingleRank) {
  CommunicatorGrid alone_grid(world, 1, 1, common::Ordering::RowMajor);

  Communicator alone_world = alone_grid.rowCommunicator();

  // just the master rank has to reduce
  if (alone_world == MPI_COMM_NULL)
    return;

  constexpr int root = 0;
  constexpr std::size_t N = 3;
  constexpr TypeParam input[N] = {TypeUtilities<TypeParam>::element(0, 1),
                                  TypeUtilities<TypeParam>::element(1, 2),
                                  TypeUtilities<TypeParam>::element(2, 3)};
  TypeParam reduced[N];

  ASSERT_EQ(alone_world.rank(), root);

  sync::reduce(root, alone_world, MPI_SUM, common::make_data(input, N), common::make_data(reduced, N));

  for (std::size_t index = 0; index < N; ++index)
    DLAF_EXPECT_NEAR(input[index], reduced[index], TypeUtilities<TypeParam>::error);
}

TEST_F(ReduceTest, Value) {
  const int root = 0;
  constexpr TypeParam value = TypeUtilities<TypeParam>::element(13, 26);
  TypeParam result = 0;

  sync::reduce(root, world, MPI_SUM, common::make_data(&value, 1), common::make_data(&result, 1));

  if (world.rank() == root)
    DLAF_EXPECT_NEAR(value * static_cast<TypeParam>(NUM_MPI_RANKS), result,
                     NUM_MPI_RANKS * TypeUtilities<TypeParam>::error);
}

TEST_F(ReduceTest, CArray) {
  static_assert(NUM_MPI_RANKS >= 2, "This test requires at least two ranks");
  int root = 1;

  constexpr std::size_t N = 3;
  constexpr TypeParam input[N] = {TypeUtilities<TypeParam>::element(0, 1),
                                  TypeUtilities<TypeParam>::element(1, 2),
                                  TypeUtilities<TypeParam>::element(2, 3)};
  TypeParam reduced[N];

  sync::reduce(root, world, MPI_SUM, common::make_data(input, N), common::make_data(reduced, N));

  if (world.rank() == root) {
    for (std::size_t index = 0; index < N; ++index)
      DLAF_EXPECT_NEAR(input[index] * static_cast<TypeParam>(NUM_MPI_RANKS), reduced[index],
                       NUM_MPI_RANKS * TypeUtilities<TypeParam>::error);
  }
}

TEST_F(ReduceTest, ContiguousToContiguous) {
  static_assert(NUM_MPI_RANKS >= 2, "This test requires at least two ranks");

  constexpr int N = 10;
  constexpr TypeParam value = TypeUtilities<TypeParam>::element(13, 26);

  TypeParam data_A[N];
  TypeParam data_B[N];

  for (auto i = 0; i < N; ++i)
    data_A[i] = value;

  constexpr int root = 0;
  auto& communicator = world;
  auto&& message_input = common::make_data(static_cast<const TypeParam*>(data_A), N);
  auto&& message_output = common::make_data(data_B, N);
  MPI_Op op = MPI_SUM;

  sync::reduce(root, communicator, op, std::move(message_input), std::move(message_output));

  if (root == world.rank()) {
    for (auto i = 0; i < N; ++i)
      DLAF_EXPECT_NEAR(value * static_cast<TypeParam>(NUM_MPI_RANKS), data_B[i],
                       NUM_MPI_RANKS * TypeUtilities<TypeParam>::error);
  }
}

TEST_F(ReduceTest, StridedToContiguous) {
  static_assert(NUM_MPI_RANKS >= 2, "This test requires at least two ranks");

  // 3 blocks, 2 elements each, with a distance of 5 elements between start of each block
  // E E - - - E E - - - E E    (without padding at the end)
  constexpr std::size_t nblocks = 3;
  constexpr std::size_t block_size = 2;
  constexpr std::size_t block_distance = 5;

  constexpr std::size_t memory_footprint = (nblocks - 1) * block_distance + block_size;
  TypeParam data_strided[memory_footprint];

  constexpr int N = nblocks * block_size;
  TypeParam data_contiguous[N];

  constexpr TypeParam value = TypeUtilities<TypeParam>::element(13, 26);
  for (std::size_t i_block = 0; i_block < nblocks; ++i_block)
    for (std::size_t i_element = 0; i_element < block_size; ++i_element) {
      auto mem_pos = i_block * block_distance + i_element;
      data_strided[mem_pos] = value;
    }

  auto&& message_input = common::make_data(static_cast<const TypeParam*>(data_strided), nblocks,
                                           block_size, block_distance);
  auto&& message_output = common::make_data(data_contiguous, N);

  constexpr int root = 0;
  auto& communicator = world;
  MPI_Op op = MPI_SUM;
  sync::reduce(root, communicator, op, std::move(message_input), std::move(message_output));

  if (root == world.rank()) {
    for (auto i = 0; i < N; ++i)
      DLAF_EXPECT_NEAR(value * static_cast<TypeParam>(NUM_MPI_RANKS), data_contiguous[i],
                       NUM_MPI_RANKS * TypeUtilities<TypeParam>::error);
  }
}

TEST_F(ReduceTest, ContiguousToStrided) {
  static_assert(NUM_MPI_RANKS >= 2, "This test requires at least two ranks");

  // 3 blocks, 2 elements each, with a distance of 5 elements between start of each block
  // E E - - - E E - - - E E    (without padding at the end)
  constexpr std::size_t nblocks = 3;
  constexpr std::size_t block_size = 2;
  constexpr std::size_t block_distance = 5;

  constexpr std::size_t memory_footprint = (nblocks - 1) * block_distance + block_size;
  TypeParam data_strided[memory_footprint];

  constexpr int N = nblocks * block_size;
  TypeParam data_contiguous[N];

  constexpr TypeParam value = TypeUtilities<TypeParam>::element(13, 26);
  for (auto i = 0; i < N; ++i)
    data_contiguous[i] = value;

  auto&& message_input = common::make_data(static_cast<const TypeParam*>(data_contiguous), N);
  auto&& message_output = common::make_data(data_strided, nblocks, block_size, block_distance);

  constexpr int root = 0;
  auto& communicator = world;
  MPI_Op op = MPI_SUM;
  sync::reduce(root, communicator, op, std::move(message_input), std::move(message_output));

  if (world.rank() == root) {
    for (std::size_t i_block = 0; i_block < nblocks; ++i_block)
      for (std::size_t i_element = 0; i_element < block_size; ++i_element) {
        auto mem_pos = i_block * block_distance + i_element;
        DLAF_EXPECT_NEAR(value * static_cast<TypeParam>(NUM_MPI_RANKS), data_strided[mem_pos],
                         NUM_MPI_RANKS * TypeUtilities<TypeParam>::error);
      }
  }
}
