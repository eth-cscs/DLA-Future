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

#include <memory>
#include <type_traits>

#include <gtest/gtest.h>

#include "dlaf/types.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;

template <typename T>
struct memory_data {
  std::unique_ptr<T[]> data;
  std::size_t num_blocks;
  std::size_t block_size;
  std::size_t stride;

  T& operator[](std::size_t index) {
    auto i_block = index / block_size;
    auto i_element = index % block_size;
    return data.get()[i_block * stride + i_element];
  }
};

template <class T>
memory_data<T> create_memory(const std::size_t num_blocks, const std::size_t blocksize,
                             const std::size_t stride) {
  assert(num_blocks > 0);
  assert(blocksize <= stride || stride == 0);

  if (num_blocks == 1)
    return {std::make_unique<T[]>(blocksize), num_blocks, blocksize, stride};

  // the last element does not have additional padding
  // no additional padding to the next (non-existing) element
  auto distance = std::max(blocksize, stride);
  auto memory_footprint = (num_blocks - 1) * distance + blocksize;
  return {std::make_unique<T[]>(memory_footprint), num_blocks, blocksize, stride};
}

enum class MEMORY_TYPE { ARRAY_CONTIGUOUS, ARRAY_STRIDED, ARRAY_CONTIGUOUS_AS_STRIDED };

template <class T>
memory_data<T> create_memory(MEMORY_TYPE type) {
  switch (type) {
    case MEMORY_TYPE::ARRAY_CONTIGUOUS:
      // 1 block
      // 13 elements
      // E E E E E E E E E E E E E
      return create_memory<T>(1, 13, 0);
    case MEMORY_TYPE::ARRAY_STRIDED:
      // 3 blocks
      // 2 elements each
      // 5 elements between start of each block
      // E E - - - E E - - - E E    (without padding at the end)
      return create_memory<T>(3, 2, 5);
    case MEMORY_TYPE::ARRAY_CONTIGUOUS_AS_STRIDED:
      // 3 blocks
      // 5 elements each
      // 5 elements between start of each block
      // E E E E E E E E E E E E E E E
      return create_memory<T>(3, 5, 5);
    default:
      throw std::runtime_error("Unknown memory type");
  }
}

template <class Type>
class BufferBasicTest : public ::testing::Test {};

TYPED_TEST_SUITE(BufferBasicTest, dlaf_test::BufferTypes);

TYPED_TEST(BufferBasicTest, MakeFromPointer) {
  TypeParam value = 26;
  TypeParam* value_ptr = &value;

  auto buffer = common::make_buffer(value_ptr, 1);

  EXPECT_EQ(value_ptr, buffer_pointer(buffer));
  EXPECT_EQ(1, buffer_nblocks(buffer));
  EXPECT_EQ(1, buffer_blocksize(buffer));
  EXPECT_EQ(0, buffer_stride(buffer));
  EXPECT_TRUE(buffer_iscontiguous(buffer));

  static_assert(common::is_buffer<decltype(buffer)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam,
                             typename common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, MakeFromPointerConst) {
  TypeParam value = 26;
  const TypeParam* value_ptr = &value;

  auto buffer = common::make_buffer(value_ptr, 1);

  EXPECT_EQ(value_ptr, buffer_pointer(buffer));
  EXPECT_EQ(1, buffer_nblocks(buffer));
  EXPECT_EQ(1, buffer_blocksize(buffer));
  EXPECT_EQ(0, buffer_stride(buffer));
  EXPECT_TRUE(buffer_iscontiguous(buffer));

  static_assert(common::is_buffer<decltype(buffer)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<const TypeParam,
                             typename common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, MakeFromCArray) {
  const int N = 13;
  TypeParam value_array[N]{};

  auto buffer = common::make_buffer(value_array, N);

  EXPECT_EQ(&value_array[0], buffer_pointer(buffer));
  EXPECT_EQ(1, buffer_nblocks(buffer));
  EXPECT_EQ(std::extent<decltype(value_array)>::value, buffer_blocksize(buffer));
  EXPECT_EQ(0, buffer_stride(buffer));
  EXPECT_TRUE(buffer_iscontiguous(buffer));

  static_assert(common::is_buffer<decltype(buffer)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam,
                             typename common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, MakeFromCArrayConst) {
  const int N = 13;
  const TypeParam value_array[N]{};

  auto buffer = common::make_buffer(value_array, N);

  EXPECT_EQ(&value_array[0], buffer_pointer(buffer));
  EXPECT_EQ(1, buffer_nblocks(buffer));
  EXPECT_EQ(std::extent<decltype(value_array)>::value, buffer_blocksize(buffer));
  EXPECT_EQ(0, buffer_stride(buffer));
  EXPECT_TRUE(buffer_iscontiguous(buffer));

  static_assert(common::is_buffer<decltype(buffer)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<const TypeParam,
                             typename common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, MakeFromContiguousArray) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS);

  auto buffer =
      common::make_buffer(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), buffer_pointer(buffer));
  EXPECT_EQ(memory.num_blocks, buffer_nblocks(buffer));
  EXPECT_EQ(memory.block_size, buffer_blocksize(buffer));
  EXPECT_EQ(memory.stride, buffer_stride(buffer));
  EXPECT_TRUE(buffer_iscontiguous(buffer));

  static_assert(common::is_buffer<decltype(buffer)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam,
                             typename common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, MakeFromContiguousArrayConst) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS);

  auto buffer = common::make_buffer(const_cast<const TypeParam*>(memory.data.get()), memory.num_blocks,
                                    memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), buffer_pointer(buffer));
  EXPECT_EQ(memory.num_blocks, buffer_nblocks(buffer));
  EXPECT_EQ(memory.block_size, buffer_blocksize(buffer));
  EXPECT_EQ(memory.stride, buffer_stride(buffer));
  EXPECT_TRUE(buffer_iscontiguous(buffer));

  static_assert(common::is_buffer<decltype(buffer)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<const TypeParam,
                             typename common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, MakeFromContiguousAsStridedArray) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS_AS_STRIDED);

  auto buffer =
      common::make_buffer(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), buffer_pointer(buffer));
  EXPECT_EQ(1, buffer_nblocks(buffer));
  EXPECT_EQ(memory.num_blocks * memory.block_size, buffer_blocksize(buffer));
  EXPECT_EQ(0, buffer_stride(buffer));
  EXPECT_TRUE(buffer_iscontiguous(buffer));

  static_assert(common::is_buffer<decltype(buffer)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam,
                             typename common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, MakeFromContiguousAsStridedArrayConst) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS_AS_STRIDED);

  auto buffer = common::make_buffer(const_cast<const TypeParam*>(memory.data.get()), memory.num_blocks,
                                    memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), buffer_pointer(buffer));
  EXPECT_EQ(1, buffer_nblocks(buffer));
  EXPECT_EQ(memory.num_blocks * memory.block_size, buffer_blocksize(buffer));
  EXPECT_EQ(0, buffer_stride(buffer));
  EXPECT_TRUE(buffer_iscontiguous(buffer));

  EXPECT_TRUE(buffer_iscontiguous(buffer));

  static_assert(common::is_buffer<decltype(buffer)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<const TypeParam,
                             typename common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, MakeFromStridedArray) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_STRIDED);

  auto buffer =
      common::make_buffer(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), buffer_pointer(buffer));
  EXPECT_EQ(memory.num_blocks, buffer_nblocks(buffer));
  EXPECT_EQ(memory.block_size, buffer_blocksize(buffer));
  EXPECT_EQ(memory.stride, buffer_stride(buffer));
  EXPECT_FALSE(buffer_iscontiguous(buffer));

  static_assert(common::is_buffer<decltype(buffer)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam,
                             typename common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, MakeFromStridedArrayConst) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_STRIDED);

  auto buffer = common::make_buffer(const_cast<const TypeParam*>(memory.data.get()), memory.num_blocks,
                                    memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), buffer_pointer(buffer));
  EXPECT_EQ(memory.num_blocks, buffer_nblocks(buffer));
  EXPECT_EQ(memory.block_size, buffer_blocksize(buffer));
  EXPECT_EQ(memory.stride, buffer_stride(buffer));
  EXPECT_FALSE(buffer_iscontiguous(buffer));

  static_assert(common::is_buffer<decltype(buffer)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<const TypeParam,
                             typename common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, MakeBufferUniquePtr) {
  const std::size_t N = 13;
  auto buffer = common::BufferWithMemory<TypeParam>(N);

  EXPECT_NE(nullptr, buffer_pointer(buffer));
  EXPECT_EQ(1, buffer_nblocks(buffer));
  EXPECT_EQ(N, buffer_blocksize(buffer));
  EXPECT_EQ(0, buffer_stride(buffer));
  EXPECT_TRUE(buffer_iscontiguous(buffer));
}

TYPED_TEST(BufferBasicTest, CtorFromPointer) {
  TypeParam value = 26;
  TypeParam* value_ptr = &value;

  auto buffer = common::BufferBasic<TypeParam>{value_ptr, 1};

  EXPECT_EQ(value_ptr, buffer_pointer(buffer));
  EXPECT_EQ(1, buffer_nblocks(buffer));
  EXPECT_EQ(1, buffer_blocksize(buffer));
  EXPECT_EQ(0, buffer_stride(buffer));
  EXPECT_TRUE(buffer_iscontiguous(buffer));

  static_assert(common::is_buffer<decltype(buffer)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam,
                             typename common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, CtorFromPointerConst) {
  TypeParam value = 26;
  const TypeParam* value_ptr = &value;

  auto buffer = common::BufferBasic<const TypeParam>{value_ptr, 1};

  EXPECT_EQ(value_ptr, buffer_pointer(buffer));
  EXPECT_EQ(1, buffer_nblocks(buffer));
  EXPECT_EQ(1, buffer_blocksize(buffer));
  EXPECT_EQ(0, buffer_stride(buffer));
  EXPECT_TRUE(buffer_iscontiguous(buffer));

  static_assert(common::is_buffer<decltype(buffer)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<const TypeParam,
                             typename common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, CtorFromCArray) {
  const int N = 13;
  TypeParam value_array[N]{};

  auto buffer = common::BufferBasic<decltype(value_array)>{value_array};

  EXPECT_EQ(&value_array[0], buffer_pointer(buffer));
  EXPECT_EQ(1, buffer_nblocks(buffer));
  EXPECT_EQ(std::extent<decltype(value_array)>::value, buffer_blocksize(buffer));
  EXPECT_EQ(0, buffer_stride(buffer));
  EXPECT_TRUE(buffer_iscontiguous(buffer));

  static_assert(common::is_buffer<decltype(buffer)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam,
                             typename common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, CtorFromCArrayConst) {
  const int N = 13;
  const TypeParam value_array[N]{};

  auto buffer = common::BufferBasic<decltype(value_array)>{value_array};

  EXPECT_EQ(&value_array[0], buffer_pointer(buffer));
  EXPECT_EQ(1, buffer_nblocks(buffer));
  EXPECT_EQ(std::extent<decltype(value_array)>::value, buffer_blocksize(buffer));
  EXPECT_EQ(0, buffer_stride(buffer));
  EXPECT_TRUE(buffer_iscontiguous(buffer));

  static_assert(common::is_buffer<decltype(buffer)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<const TypeParam,
                             typename common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, CtorFromContiguousArray) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS);

  auto buffer =
      common::make_buffer(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), buffer_pointer(buffer));
  EXPECT_EQ(memory.num_blocks, buffer_nblocks(buffer));
  EXPECT_EQ(memory.block_size, buffer_blocksize(buffer));
  EXPECT_EQ(memory.stride, buffer_stride(buffer));
  EXPECT_TRUE(buffer_iscontiguous(buffer));

  static_assert(common::is_buffer<decltype(buffer)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam,
                             typename common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, CtorFromContiguousArrayConst) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS);

  auto buffer = common::make_buffer(const_cast<const TypeParam*>(memory.data.get()), memory.num_blocks,
                                    memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), buffer_pointer(buffer));
  EXPECT_EQ(memory.num_blocks, buffer_nblocks(buffer));
  EXPECT_EQ(memory.block_size, buffer_blocksize(buffer));
  EXPECT_EQ(memory.stride, buffer_stride(buffer));
  EXPECT_TRUE(buffer_iscontiguous(buffer));

  static_assert(common::is_buffer<decltype(buffer)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<const TypeParam,
                             typename common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, CtorFromContiguousAsStridedArray) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS_AS_STRIDED);

  auto buffer =
      common::make_buffer(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), buffer_pointer(buffer));
  EXPECT_EQ(1, buffer_nblocks(buffer));
  EXPECT_EQ(memory.num_blocks * memory.block_size, buffer_blocksize(buffer));
  EXPECT_EQ(0, buffer_stride(buffer));
  EXPECT_TRUE(buffer_iscontiguous(buffer));

  static_assert(common::is_buffer<decltype(buffer)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam,
                             typename common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, CtorFromContiguousAsStridedArrayConst) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS_AS_STRIDED);

  auto buffer = common::make_buffer(const_cast<const TypeParam*>(memory.data.get()), memory.num_blocks,
                                    memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), buffer_pointer(buffer));
  EXPECT_EQ(1, buffer_nblocks(buffer));
  EXPECT_EQ(memory.num_blocks * memory.block_size, buffer_blocksize(buffer));
  EXPECT_EQ(0, buffer_stride(buffer));
  EXPECT_TRUE(buffer_iscontiguous(buffer));

  static_assert(common::is_buffer<decltype(buffer)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<const TypeParam,
                             typename common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, CtorFromStridedArray) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_STRIDED);

  auto buffer =
      common::make_buffer(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), buffer_pointer(buffer));
  EXPECT_EQ(memory.num_blocks, buffer_nblocks(buffer));
  EXPECT_EQ(memory.block_size, buffer_blocksize(buffer));
  EXPECT_EQ(memory.stride, buffer_stride(buffer));
  EXPECT_FALSE(buffer_iscontiguous(buffer));

  static_assert(common::is_buffer<decltype(buffer)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam,
                             typename common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, CtorFromStridedArrayConst) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_STRIDED);

  auto buffer = common::make_buffer(const_cast<const TypeParam*>(memory.data.get()), memory.num_blocks,
                                    memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), buffer_pointer(buffer));
  EXPECT_EQ(memory.num_blocks, buffer_nblocks(buffer));
  EXPECT_EQ(memory.block_size, buffer_blocksize(buffer));
  EXPECT_EQ(memory.stride, buffer_stride(buffer));
  EXPECT_FALSE(buffer_iscontiguous(buffer));

  static_assert(common::is_buffer<decltype(buffer)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<const TypeParam,
                             typename common::buffer_traits<decltype(buffer)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, CtorBufferUniquePtr) {
  const std::size_t N = 13;
  auto buffer = common::BufferWithMemory<TypeParam>(N);

  EXPECT_NE(nullptr, buffer_pointer(buffer));
  EXPECT_EQ(1, buffer_nblocks(buffer));
  EXPECT_EQ(N, buffer_blocksize(buffer));
  EXPECT_EQ(0, buffer_stride(buffer));
  EXPECT_TRUE(buffer_iscontiguous(buffer));
}

template <class TypeParam, class Buffer>
void check_copy_ctor(Buffer& buffer) {
  auto buffer_copy = buffer;

  EXPECT_EQ(buffer_pointer(buffer), buffer_pointer(buffer_copy));
  EXPECT_EQ(buffer_nblocks(buffer), buffer_nblocks(buffer_copy));
  EXPECT_EQ(buffer_blocksize(buffer), buffer_blocksize(buffer_copy));
  EXPECT_EQ(buffer_stride(buffer), buffer_stride(buffer_copy));
  EXPECT_EQ(buffer_iscontiguous(buffer), buffer_iscontiguous(buffer_copy));

  static_assert(common::is_buffer<decltype(buffer_copy)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam,
                             typename common::buffer_traits<decltype(buffer_copy)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(BufferBasicTest, CopyCtorFromPointer) {
  TypeParam value = 26;
  auto buffer = common::make_buffer(&value, 1);
  check_copy_ctor<TypeParam>(buffer);

  const TypeParam value_const = value;
  auto buffer_const = common::make_buffer(&value_const, 1);
  check_copy_ctor<const TypeParam>(buffer_const);
}

TYPED_TEST(BufferBasicTest, CopyCtorFromCArray) {
  const int N = 13;
  TypeParam value_array[N]{};

  auto buffer = common::make_buffer(value_array, N);
  check_copy_ctor<TypeParam>(buffer);

  const TypeParam value_array_const[N]{};
  auto buffer_const = common::make_buffer(value_array_const, N);
  check_copy_ctor<const TypeParam>(buffer_const);
}

TYPED_TEST(BufferBasicTest, CopyCtorFromContiguousArray) {
  auto memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS);

  auto buffer =
      common::make_buffer(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);
  check_copy_ctor<TypeParam>(buffer);

  auto buffer_const = common::make_buffer(static_cast<const TypeParam*>(memory.data.get()),
                                          memory.num_blocks, memory.block_size, memory.stride);
  check_copy_ctor<const TypeParam>(buffer_const);
}

TYPED_TEST(BufferBasicTest, CopyCtorFromContiguousAsStridedArray) {
  auto memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS_AS_STRIDED);

  auto buffer =
      common::make_buffer(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);
  check_copy_ctor<TypeParam>(buffer);

  auto buffer_const = common::make_buffer(static_cast<const TypeParam*>(memory.data.get()),
                                          memory.num_blocks, memory.block_size, memory.stride);
  check_copy_ctor<const TypeParam>(buffer_const);
}

TYPED_TEST(BufferBasicTest, CopyCtorFromStridedArray) {
  auto memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_STRIDED);

  auto buffer =
      common::make_buffer(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);
  check_copy_ctor<TypeParam>(buffer);

  auto buffer_const = common::make_buffer(const_cast<const TypeParam*>(memory.data.get()),
                                          memory.num_blocks, memory.block_size, memory.stride);
  check_copy_ctor<const TypeParam>(buffer_const);
}

template <class TypeParam, class Buffer>
void check_temporary(Buffer& buffer) {
  auto buffer_temp = create_temporary_buffer(buffer);

  EXPECT_NE(common::buffer_pointer(buffer), common::buffer_pointer(buffer_temp));
  EXPECT_EQ(common::buffer_count(buffer), common::buffer_count(buffer_temp));
  EXPECT_TRUE(common::buffer_iscontiguous(buffer_temp));
}

TYPED_TEST(BufferBasicTest, CreateTemporaryFromPointer) {
  TypeParam value = 26;
  auto buffer = common::make_buffer(&value, 1);
  check_temporary<TypeParam>(buffer);

  const TypeParam value_const = value;
  auto buffer_const = common::make_buffer(&value_const, 1);
  check_temporary<const TypeParam>(buffer_const);
}

TYPED_TEST(BufferBasicTest, CreateTemporaryFromCArray) {
  const int N = 13;
  TypeParam value_array[N]{};

  auto buffer = common::make_buffer(value_array, N);
  check_temporary<TypeParam>(buffer);

  const TypeParam value_array_const[N]{};
  auto buffer_const = common::make_buffer(value_array_const, N);
  check_temporary<const TypeParam>(buffer_const);
}

TYPED_TEST(BufferBasicTest, CreateTemporaryFromContiguousArray) {
  auto memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS);

  auto buffer =
      common::make_buffer(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);
  check_temporary<TypeParam>(buffer);

  auto buffer_const = common::make_buffer(static_cast<const TypeParam*>(memory.data.get()),
                                          memory.num_blocks, memory.block_size, memory.stride);
  check_temporary<const TypeParam>(buffer_const);
}

TYPED_TEST(BufferBasicTest, CreateTemporaryFromContiguousAsStridedArray) {
  auto memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS_AS_STRIDED);

  auto buffer =
      common::make_buffer(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);
  check_temporary<TypeParam>(buffer);

  auto buffer_const = common::make_buffer(static_cast<const TypeParam*>(memory.data.get()),
                                          memory.num_blocks, memory.block_size, memory.stride);
  check_temporary<const TypeParam>(buffer_const);
}

TYPED_TEST(BufferBasicTest, CreateTemporaryFromStridedArray) {
  auto memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_STRIDED);

  auto buffer =
      common::make_buffer(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);
  check_temporary<TypeParam>(buffer);

  auto buffer_const = common::make_buffer(const_cast<const TypeParam*>(memory.data.get()),
                                          memory.num_blocks, memory.block_size, memory.stride);
  check_temporary<const TypeParam>(buffer_const);
}

template <class T>
auto create_buffer_from_memory(memory_data<T>& memory) {
  return common::make_buffer(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);
}

template <class T>
auto create_const_buffer_from_memory(memory_data<T>& memory) {
  return common::make_buffer(const_cast<const T*>(memory.data.get()), memory.num_blocks,
                             memory.block_size, memory.stride);
}

TYPED_TEST(BufferBasicTest, CopyDataCArrays) {
  const int N = 13;
  TypeParam memory_src[N];
  TypeParam memory_dst[N];

  for (int i = 0; i < N; ++i)
    memory_src[i] = dlaf_test::TypeUtilities<TypeParam>::element(i, 0);

  auto buffer_src = common::make_buffer(const_cast<const TypeParam*>(memory_src), N);
  auto buffer_dest = common::make_buffer(memory_dst, N);

  common::copy(buffer_src, buffer_dest);

  for (int i = 0; i < N; ++i)
    EXPECT_EQ(dlaf_test::TypeUtilities<TypeParam>::element(i, 0), memory_dst[i]);
}

TYPED_TEST(BufferBasicTest, CopyDataArrays) {
  auto memory_types = {MEMORY_TYPE::ARRAY_CONTIGUOUS, MEMORY_TYPE::ARRAY_CONTIGUOUS_AS_STRIDED,
                       MEMORY_TYPE::ARRAY_STRIDED};

  for (auto memory_type : memory_types) {
    auto memory_src = create_memory<TypeParam>(memory_type);
    auto memory_dest = create_memory<TypeParam>(memory_type);

    for (std::size_t i = 0; i < memory_src.num_blocks * memory_src.block_size; ++i)
      memory_src[i] = dlaf_test::TypeUtilities<TypeParam>::element(i, 0);

    auto buffer_src = create_const_buffer_from_memory(memory_src);
    auto buffer_dest = create_buffer_from_memory(memory_dest);

    common::copy(buffer_src, buffer_dest);

    for (std::size_t i = 0; i < memory_src.num_blocks * memory_src.block_size; ++i)
      EXPECT_EQ(dlaf_test::TypeUtilities<TypeParam>::element(i, 0), memory_dest[i]);
  }
}

TYPED_TEST(BufferBasicTest, CopyDataHeterogeneous) {
  const std::size_t N = 26;
  const std::size_t N_GROUPS = 2;
  static_assert(N % N_GROUPS == 0, "Incompatible geometry");

  std::vector<memory_data<TypeParam>> memory_types;

  // memory contiguous
  memory_types.emplace_back(create_memory<TypeParam>(1, N, 0));
  // memory contiguous as strided
  memory_types.emplace_back(create_memory<TypeParam>(N_GROUPS, N / N_GROUPS, N / N_GROUPS));
  // memory strided
  memory_types.emplace_back(create_memory<TypeParam>(N_GROUPS, N / N_GROUPS, N / N_GROUPS + 5));

  // CArray as source
  for (auto& memory_dest : memory_types) {
    TypeParam memory_array[N];
    for (std::size_t i = 0; i < N; ++i)
      memory_array[i] = dlaf_test::TypeUtilities<TypeParam>::element(i, 0);

    for (std::size_t i = 0; i < N; ++i)
      memory_dest[i] = 0;

    auto buffer_array = common::make_buffer(memory_array, N);
    auto buffer_dest = create_buffer_from_memory(memory_dest);

    copy(buffer_array, buffer_dest);

    for (std::size_t i = 0; i < N; ++i)
      EXPECT_EQ(dlaf_test::TypeUtilities<TypeParam>::element(i, 0), memory_dest[i]);
  }

  // CArray as destination
  for (auto& memory_src : memory_types) {
    TypeParam memory_array[N];
    for (std::size_t i = 0; i < N; ++i)
      memory_array[i] = 0;

    for (std::size_t i = 0; i < N; ++i)
      memory_src[i] = dlaf_test::TypeUtilities<TypeParam>::element(i, 0);

    auto buffer_src = create_const_buffer_from_memory(memory_src);
    auto buffer_array = common::make_buffer(memory_array, N);

    copy(buffer_src, buffer_array);

    for (std::size_t i = 0; i < N; ++i)
      EXPECT_EQ(dlaf_test::TypeUtilities<TypeParam>::element(i, 0), memory_array[i]);
  }

  // Other combinations
  for (auto& memory_src : memory_types) {
    for (auto& memory_dest : memory_types) {
      if (&memory_src == &memory_dest)
        continue;

      for (std::size_t i = 0; i < N; ++i)
        memory_src[i] = dlaf_test::TypeUtilities<TypeParam>::element(i, 0);

      for (std::size_t i = 0; i < N; ++i)
        memory_dest[i] = 0;

      auto buffer_src = create_const_buffer_from_memory(memory_src);
      auto buffer_dest = create_buffer_from_memory(memory_dest);

      copy(buffer_src, buffer_dest);

      for (std::size_t i = 0; i < N; ++i)
        EXPECT_EQ(dlaf_test::TypeUtilities<TypeParam>::element(i, 0), memory_dest[i]);
    }
  }
}
