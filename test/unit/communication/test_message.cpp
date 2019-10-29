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

#include "dlaf/types.h"
#include "dlaf_test/util_types.h"

template <class Type>
class MessageTest : public ::testing::Test {};

TYPED_TEST_CASE(MessageTest, dlaf_test::BufferTypes);

TYPED_TEST(MessageTest, MakeFromPointer) {
  using namespace dlaf::common;

  TypeParam value = 26;
  TypeParam* value_ptr = &value;

  auto buffer = make_buffer(value_ptr, 1);

  auto message_direct = dlaf::comm::make_message(make_buffer(value_ptr, 1));
  auto message_indirect = dlaf::comm::make_message(value_ptr, 1);

  int type_direct_size;
  MPI_Type_size(message_direct.mpi_type(), &type_direct_size);
  int type_indirect_size;
  MPI_Type_size(message_indirect.mpi_type(), &type_indirect_size);

  EXPECT_EQ(value_ptr, message_direct.data());
  EXPECT_EQ(message_direct.data(), message_indirect.data());

  EXPECT_EQ(1, message_direct.count());
  EXPECT_EQ(1, message_indirect.count());

  EXPECT_EQ(sizeof(TypeParam), type_direct_size);
  EXPECT_EQ(sizeof(TypeParam), type_indirect_size);

  EXPECT_EQ(static_cast<MPI_Datatype>(dlaf::comm::mpi_datatype<TypeParam>::type),
            message_direct.mpi_type());
  EXPECT_EQ(static_cast<MPI_Datatype>(dlaf::comm::mpi_datatype<TypeParam>::type),
            message_indirect.mpi_type());

  static_assert(std::is_same<TypeParam, typename decltype(message_direct)::element_t>::value,
                "Wrong type");
  static_assert(std::is_same<TypeParam, typename decltype(message_indirect)::element_t>::value,
                "Wrong type");
}

TYPED_TEST(MessageTest, MakeFromContiguousArray) {
  const int N = 13;
  TypeParam value_array[N]{};

  auto buffer = dlaf::common::make_buffer(value_array, N);

  auto message_direct = dlaf::comm::make_message(dlaf::common::make_buffer(value_array, N));
  auto message_indirect = dlaf::comm::make_message(value_array, N);

  int type_direct_size;
  MPI_Type_size(message_direct.mpi_type(), &type_direct_size);
  int type_indirect_size;
  MPI_Type_size(message_indirect.mpi_type(), &type_indirect_size);

  EXPECT_EQ(value_array, message_direct.data());
  EXPECT_EQ(message_direct.data(), message_indirect.data());

  EXPECT_EQ(N, message_direct.count());
  EXPECT_EQ(N, message_indirect.count());

  EXPECT_EQ(sizeof(TypeParam), type_direct_size);
  EXPECT_EQ(sizeof(TypeParam), type_indirect_size);

  EXPECT_EQ(static_cast<MPI_Datatype>(dlaf::comm::mpi_datatype<TypeParam>::type),
            message_direct.mpi_type());
  EXPECT_EQ(static_cast<MPI_Datatype>(dlaf::comm::mpi_datatype<TypeParam>::type),
            message_indirect.mpi_type());

  static_assert(std::is_same<TypeParam, typename decltype(message_direct)::element_t>::value,
                "Wrong type");
  static_assert(std::is_same<TypeParam, typename decltype(message_indirect)::element_t>::value,
                "Wrong type");
}

TYPED_TEST(MessageTest, MakeFromStridedArray) {
  using dlaf::SizeType;

  // 3 blocks, 2 elements each, with a distance of 5 elements between start of each block
  // E E - - - E E - - - E E    (without padding at the end)
  const SizeType nblocks = 3;
  const SizeType block_size = 2;
  const SizeType block_distance = 5;
  const SizeType total_elements = nblocks * block_size;

  const std::size_t memory_footprint = (nblocks - 1) * block_distance + block_size;
  TypeParam value_array[memory_footprint]{};

  auto buffer = dlaf::common::make_buffer(value_array, nblocks, block_size, block_distance);

  auto message_direct = dlaf::comm::make_message(
      dlaf::common::make_buffer(value_array, nblocks, block_size, block_distance));
  auto message_indirect = dlaf::comm::make_message(value_array, nblocks, block_size, block_distance);

  int type_direct_size;
  MPI_Type_size(message_direct.mpi_type(), &type_direct_size);
  int type_indirect_size;
  MPI_Type_size(message_indirect.mpi_type(), &type_indirect_size);

  EXPECT_EQ(value_array, message_direct.data());
  EXPECT_EQ(message_direct.data(), message_indirect.data());

  EXPECT_EQ(1, message_direct.count());
  EXPECT_EQ(1, message_indirect.count());

  EXPECT_EQ(total_elements * sizeof(TypeParam), type_direct_size);
  EXPECT_EQ(total_elements * sizeof(TypeParam), type_indirect_size);

  EXPECT_NE(MPI_DATATYPE_NULL, message_direct.mpi_type());
  EXPECT_NE(MPI_DATATYPE_NULL, message_indirect.mpi_type());

  static_assert(std::is_same<TypeParam, typename decltype(message_direct)::element_t>::value,
                "Wrong type");
  static_assert(std::is_same<TypeParam, typename decltype(message_indirect)::element_t>::value,
                "Wrong type");
}

TYPED_TEST(MessageTest, MoveBasicType) {
  using namespace dlaf::common;

  TypeParam value = 26;
  TypeParam* value_ptr = &value;

  auto message = dlaf::comm::make_message(value_ptr, 1);

  auto new_message = std::move(message);

  int type_size;
  MPI_Type_size(new_message.mpi_type(), &type_size);

  EXPECT_EQ(value_ptr, new_message.data());
  EXPECT_EQ(1, new_message.count());
  EXPECT_EQ(static_cast<MPI_Datatype>(new_message.mpi_type()), new_message.mpi_type());
  EXPECT_EQ(sizeof(TypeParam), type_size);

  static_assert(std::is_same<TypeParam, typename decltype(new_message)::element_t>::value, "Wrong type");
}

TYPED_TEST(MessageTest, MoveCustomType) {
  using dlaf::SizeType;

  // 3 blocks, 2 elements each, with a distance of 5 elements between start of each block
  // E E - - - E E - - - E E    (without padding at the end)
  const SizeType nblocks = 3;
  const SizeType block_size = 2;
  const SizeType block_distance = 5;
  const SizeType total_elements = nblocks * block_size;

  const std::size_t memory_footprint = (nblocks - 1) * block_distance + block_size;
  TypeParam value_array[memory_footprint]{};

  auto message = dlaf::comm::make_message(value_array, nblocks, block_size, block_distance);

  auto new_message = std::move(message);

  int type_size;
  MPI_Type_size(new_message.mpi_type(), &type_size);

  EXPECT_EQ(value_array, new_message.data());
  EXPECT_EQ(1, new_message.count());
  EXPECT_EQ(total_elements * sizeof(TypeParam), type_size);
  EXPECT_NE(MPI_DATATYPE_NULL, new_message.mpi_type());

  static_assert(std::is_same<TypeParam, typename decltype(new_message)::element_t>::value, "Wrong type");
}
