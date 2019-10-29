//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/common/buffer.h"

#include <type_traits>

#include <gtest/gtest.h>

#include "dlaf/types.h"
#include "dlaf_test/util_types.h"

template <class Type>
class BufferTest : public ::testing::Test {};

TYPED_TEST_CASE(BufferTest, dlaf_test::BufferTypes);

TYPED_TEST(BufferTest, MakeFromPointer) {
  using namespace dlaf::common;

  TypeParam value = 26;
  TypeParam* value_ptr = &value;

  auto buffer = dlaf::common::make_buffer(value_ptr, 1);

  EXPECT_EQ(value_ptr, get_pointer(buffer));
  EXPECT_EQ(1, get_num_blocks(buffer));
  EXPECT_EQ(1, get_blocksize(buffer));
  EXPECT_EQ(0, get_stride(buffer));

  static_assert(std::is_same<TypeParam*, decltype(get_pointer(buffer))>::value, "Wrong pointer type");

  static_assert(std::is_same<TypeParam,
                             typename dlaf::common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferTest, MakeFromContiguousArray) {
  const int N = 13;
  TypeParam value_array[N]{};

  auto buffer = dlaf::common::make_buffer(value_array, N);

  EXPECT_EQ(&value_array[0], get_pointer(buffer));
  EXPECT_EQ(1, get_num_blocks(buffer));
  EXPECT_EQ(std::extent<decltype(value_array)>::value, get_blocksize(buffer));
  EXPECT_EQ(0, get_stride(buffer));

  static_assert(std::is_same<TypeParam*, decltype(get_pointer(buffer))>::value, "Wrong pointer type");

  static_assert(std::is_same<TypeParam,
                             typename dlaf::common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferTest, MakeFromStridedArray) {
  using dlaf::SizeType;

  // 3 blocks, 2 elements each, with a distance of 5 elements between start of each block
  // E E - - - E E - - - E E    (without padding at the end)
  const SizeType nblocks = 3;
  const SizeType block_size = 2;
  const SizeType block_distance = 5;

  const std::size_t memory_footprint = (nblocks - 1) * block_distance + block_size;
  TypeParam value_array[memory_footprint]{};

  auto buffer = dlaf::common::make_buffer(value_array, nblocks, block_size, block_distance);

  EXPECT_EQ(&value_array[0], get_pointer(buffer));
  EXPECT_EQ(nblocks, get_num_blocks(buffer));
  EXPECT_EQ(block_size, get_blocksize(buffer));
  EXPECT_EQ(block_distance, get_stride(buffer));

  static_assert(std::is_same<TypeParam*, decltype(get_pointer(buffer))>::value, "Wrong pointer type");

  static_assert(std::is_same<TypeParam,
                             typename dlaf::common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferTest, CtorFromPointer) {
  using namespace dlaf::common;

  TypeParam value = 26;
  TypeParam* value_ptr = &value;

  auto buffer = dlaf::common::Buffer<TypeParam*>{value_ptr, 1};

  EXPECT_EQ(value_ptr, get_pointer(buffer));
  EXPECT_EQ(1, get_num_blocks(buffer));
  EXPECT_EQ(1, get_blocksize(buffer));
  EXPECT_EQ(0, get_stride(buffer));

  static_assert(std::is_same<TypeParam*, decltype(get_pointer(buffer))>::value, "Wrong pointer type");

  static_assert(std::is_same<TypeParam,
                             typename dlaf::common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferTest, CtorFromContiguousArray) {
  const int N = 13;
  TypeParam value_array[N]{};

  auto buffer = dlaf::common::Buffer<decltype(value_array)>{value_array};

  EXPECT_EQ(&value_array[0], get_pointer(buffer));
  EXPECT_EQ(1, get_num_blocks(buffer));
  EXPECT_EQ(std::extent<decltype(value_array)>::value, get_blocksize(buffer));
  EXPECT_EQ(0, get_stride(buffer));

  static_assert(std::is_same<TypeParam*, decltype(get_pointer(buffer))>::value, "Wrong pointer type");

  static_assert(std::is_same<TypeParam,
                             typename dlaf::common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferTest, CtorFromStridedArray) {
  using dlaf::SizeType;

  // 3 blocks, 2 elements each, with a distance of 5 elements between start of each block
  // E E - - - E E - - - E E    (without padding at the end)
  const SizeType nblocks = 3;
  const SizeType block_size = 2;
  const SizeType block_distance = 5;

  const std::size_t memory_footprint = (nblocks - 1) * block_distance + block_size;
  TypeParam value_array[memory_footprint]{};

  auto buffer = dlaf::common::Buffer<TypeParam*>(value_array, nblocks, block_size, block_distance);

  EXPECT_EQ(&value_array[0], get_pointer(buffer));
  EXPECT_EQ(nblocks, get_num_blocks(buffer));
  EXPECT_EQ(block_size, get_blocksize(buffer));
  EXPECT_EQ(block_distance, get_stride(buffer));

  static_assert(std::is_same<TypeParam*, decltype(get_pointer(buffer))>::value, "Wrong pointer type");

  static_assert(std::is_same<TypeParam,
                             typename dlaf::common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferTest, CopyFromPointer) {
  using namespace dlaf::common;

  TypeParam value = 26;
  TypeParam* value_ptr = &value;

  auto buffer = make_buffer(value_ptr, 1);

  auto buffer_copy = buffer;

  EXPECT_EQ(value_ptr, get_pointer(buffer_copy));
  EXPECT_EQ(1, get_num_blocks(buffer_copy));
  EXPECT_EQ(1, get_blocksize(buffer_copy));
  EXPECT_EQ(0, get_stride(buffer_copy));

  static_assert(std::is_same<TypeParam*, decltype(get_pointer(buffer_copy))>::value,
                "Wrong pointer type");

  static_assert(std::is_same<TypeParam, typename dlaf::common::buffer_traits<decltype(
                                            buffer_copy)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferTest, CopyContiguousArray) {
  const int N = 13;
  TypeParam value_array[N]{};

  auto buffer = dlaf::common::make_buffer(value_array, N);

  auto buffer_copy = buffer;

  EXPECT_EQ(&value_array[0], get_pointer(buffer_copy));
  EXPECT_EQ(1, get_num_blocks(buffer_copy));
  EXPECT_EQ(std::extent<decltype(value_array)>::value, get_blocksize(buffer_copy));
  EXPECT_EQ(0, get_stride(buffer_copy));

  static_assert(std::is_same<TypeParam*, decltype(get_pointer(buffer_copy))>::value,
                "Wrong pointer type");

  static_assert(std::is_same<TypeParam, typename dlaf::common::buffer_traits<decltype(
                                            buffer_copy)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferTest, CopyStridedArray) {
  using dlaf::SizeType;

  // 3 blocks, 2 elements each, with a distance of 5 elements between start of each block
  // E E - - - E E - - - E E    (without padding at the end)
  const SizeType nblocks = 3;
  const SizeType block_size = 2;
  const SizeType block_distance = 5;

  const std::size_t memory_footprint = (nblocks - 1) * block_distance + block_size;
  TypeParam value_array[memory_footprint]{};

  auto buffer = dlaf::common::make_buffer(value_array, nblocks, block_size, block_distance);

  auto buffer_copy = buffer;

  EXPECT_EQ(&value_array[0], get_pointer(buffer_copy));
  EXPECT_EQ(nblocks, get_num_blocks(buffer_copy));
  EXPECT_EQ(block_size, get_blocksize(buffer_copy));
  EXPECT_EQ(block_distance, get_stride(buffer_copy));

  static_assert(std::is_same<TypeParam*, decltype(get_pointer(buffer_copy))>::value,
                "Wrong pointer type");

  static_assert(std::is_same<TypeParam, typename dlaf::common::buffer_traits<decltype(
                                            buffer_copy)>::element_t>::value,
                "Wrong type returned");
}
