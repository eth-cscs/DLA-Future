//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/communication/message.h"

#include <gtest/gtest.h>

#include "dlaf/common/data_descriptor.h"
#include "dlaf/types.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf_test;
using namespace dlaf::comm;

using dlaf::common::make_data;
using dlaf::comm::make_message;

template <class Type>
class MessageTest : public ::testing::Test {};

TYPED_TEST_SUITE(MessageTest, ElementTypes);

TYPED_TEST(MessageTest, MakeFromPointer) {
  TypeParam value = 26;
  TypeParam* value_ptr = &value;

  auto data = make_data(value_ptr, 1);
  auto message = make_message(data);

  int type_size;
  MPI_Type_size(message.mpi_type(), &type_size);

  EXPECT_EQ(value_ptr, message.data());
  EXPECT_EQ(1, message.count());
  EXPECT_EQ(sizeof(TypeParam), type_size);
  EXPECT_EQ(static_cast<MPI_Datatype>(mpi_datatype<TypeParam>::type), message.mpi_type());
}

TYPED_TEST(MessageTest, MakeFromContiguousArray) {
  const int N = 13;
  TypeParam value_array[N]{};

  auto data = make_data(value_array, N);
  auto message = make_message(data);

  int type_size;
  MPI_Type_size(message.mpi_type(), &type_size);

  EXPECT_EQ(value_array, message.data());
  EXPECT_EQ(N, message.count());
  EXPECT_EQ(sizeof(TypeParam), type_size);
  EXPECT_EQ(static_cast<MPI_Datatype>(mpi_datatype<TypeParam>::type), message.mpi_type());
}

TYPED_TEST(MessageTest, MakeFromContiguousAsStridedArray) {
  // 3 blocks, 5 elements each, with a distance of 5 elements between start of each block
  // E E E E E E E E E E E E E E E     (without padding at the end)
  const SizeType nblocks = 3;
  const SizeType block_size = 5;
  const SizeType block_distance = block_size;
  const SizeType total_elements = nblocks * block_size;

  const std::size_t memory_footprint = (nblocks - 1) * block_distance + block_size;
  TypeParam value_array[memory_footprint]{};

  auto data = make_data(value_array, nblocks, block_size, block_distance);
  auto message = make_message(data);

  int type_size;
  MPI_Type_size(message.mpi_type(), &type_size);

  EXPECT_EQ(value_array, message.data());
  EXPECT_EQ(total_elements, message.count());
  EXPECT_EQ(sizeof(TypeParam), type_size);
  EXPECT_EQ(static_cast<MPI_Datatype>(mpi_datatype<TypeParam>::type), message.mpi_type());
}

TYPED_TEST(MessageTest, MakeFromStridedArray) {
  // 3 blocks, 2 elements each, with a distance of 5 elements between start of each block
  // E E - - - E E - - - E E    (without padding at the end)
  const SizeType nblocks = 3;
  const SizeType block_size = 2;
  const SizeType block_distance = 5;
  const SizeType total_elements = nblocks * block_size;

  const std::size_t memory_footprint = (nblocks - 1) * block_distance + block_size;
  TypeParam value_array[memory_footprint]{};

  auto data = make_data(value_array, nblocks, block_size, block_distance);
  auto message = make_message(data);

  int type_size;
  MPI_Type_size(message.mpi_type(), &type_size);

  EXPECT_EQ(value_array, message.data());
  EXPECT_EQ(1, message.count());
  EXPECT_EQ(total_elements * sizeof(TypeParam), type_size);
  EXPECT_NE(MPI_DATATYPE_NULL, message.mpi_type());
}

TYPED_TEST(MessageTest, MoveBasicType) {
  TypeParam value = 26;
  TypeParam* value_ptr = &value;

  auto message = make_message(common::make_data(value_ptr, 1));

  auto new_message = std::move(message);

  int type_size;
  MPI_Type_size(new_message.mpi_type(), &type_size);

  EXPECT_EQ(value_ptr, new_message.data());
  EXPECT_EQ(1, new_message.count());
  EXPECT_EQ(static_cast<MPI_Datatype>(new_message.mpi_type()), new_message.mpi_type());
  EXPECT_EQ(sizeof(TypeParam), type_size);
}

TYPED_TEST(MessageTest, MoveCustomType) {
  // 3 blocks, 2 elements each, with a distance of 5 elements between start of each block
  // E E - - - E E - - - E E    (without padding at the end)
  const SizeType nblocks = 3;
  const SizeType block_size = 2;
  const SizeType block_distance = 5;
  const SizeType total_elements = nblocks * block_size;

  const std::size_t memory_footprint = (nblocks - 1) * block_distance + block_size;
  TypeParam value_array[memory_footprint]{};

  auto message = make_message(common::make_data(value_array, nblocks, block_size, block_distance));

  auto new_message = std::move(message);

  int type_size;
  MPI_Type_size(new_message.mpi_type(), &type_size);

  EXPECT_EQ(value_array, new_message.data());
  EXPECT_EQ(1, new_message.count());
  EXPECT_EQ(total_elements * sizeof(TypeParam), type_size);
  EXPECT_NE(MPI_DATATYPE_NULL, new_message.mpi_type());
}
