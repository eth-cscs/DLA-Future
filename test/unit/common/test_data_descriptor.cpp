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
  DLAF_ASSERT_HEAVY((num_blocks > 0));
  DLAF_ASSERT_HEAVY((blocksize <= stride || stride == 0));

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
class DataDescriptorTest : public ::testing::Test {};

TYPED_TEST_SUITE(DataDescriptorTest, dlaf_test::ElementTypes);

TYPED_TEST(DataDescriptorTest, MakeFromPointer) {
  TypeParam value = 26;
  TypeParam* value_ptr = &value;

  auto data = common::make_data(value_ptr, 1);

  EXPECT_EQ(value_ptr, data_pointer(data));
  EXPECT_EQ(1, data_nblocks(data));
  EXPECT_EQ(1, data_blocksize(data));
  EXPECT_EQ(0, data_stride(data));
  EXPECT_TRUE(data_iscontiguous(data));

  static_assert(common::is_data<decltype(data)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam, typename common::data_traits<decltype(data)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, MakeFromPointerConst) {
  TypeParam value = 26;
  const TypeParam* value_ptr = &value;

  auto data = common::make_data(value_ptr, 1);

  EXPECT_EQ(value_ptr, data_pointer(data));
  EXPECT_EQ(1, data_nblocks(data));
  EXPECT_EQ(1, data_blocksize(data));
  EXPECT_EQ(0, data_stride(data));
  EXPECT_TRUE(data_iscontiguous(data));

  static_assert(common::is_data<decltype(data)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<const TypeParam,
                             typename common::data_traits<decltype(data)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, MakeFromCArray) {
  const int N = 13;
  TypeParam value_array[N]{};

  auto data = common::make_data(value_array, N);

  EXPECT_EQ(&value_array[0], data_pointer(data));
  EXPECT_EQ(1, data_nblocks(data));
  EXPECT_EQ(std::extent<decltype(value_array)>::value, data_blocksize(data));
  EXPECT_EQ(0, data_stride(data));
  EXPECT_TRUE(data_iscontiguous(data));

  static_assert(common::is_data<decltype(data)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam, typename common::data_traits<decltype(data)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, MakeFromCArrayConst) {
  const int N = 13;
  const TypeParam value_array[N]{};

  auto data = common::make_data(value_array, N);

  EXPECT_EQ(&value_array[0], data_pointer(data));
  EXPECT_EQ(1, data_nblocks(data));
  EXPECT_EQ(std::extent<decltype(value_array)>::value, data_blocksize(data));
  EXPECT_EQ(0, data_stride(data));
  EXPECT_TRUE(data_iscontiguous(data));

  static_assert(common::is_data<decltype(data)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<const TypeParam,
                             typename common::data_traits<decltype(data)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, MakeFromContiguousArray) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS);

  auto data = common::make_data(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), data_pointer(data));
  EXPECT_EQ(memory.num_blocks, data_nblocks(data));
  EXPECT_EQ(memory.block_size, data_blocksize(data));
  EXPECT_EQ(memory.stride, data_stride(data));
  EXPECT_TRUE(data_iscontiguous(data));

  static_assert(common::is_data<decltype(data)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam, typename common::data_traits<decltype(data)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, MakeFromContiguousArrayConst) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS);

  auto data = common::make_data(const_cast<const TypeParam*>(memory.data.get()), memory.num_blocks,
                                memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), data_pointer(data));
  EXPECT_EQ(memory.num_blocks, data_nblocks(data));
  EXPECT_EQ(memory.block_size, data_blocksize(data));
  EXPECT_EQ(memory.stride, data_stride(data));
  EXPECT_TRUE(data_iscontiguous(data));

  static_assert(common::is_data<decltype(data)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<const TypeParam,
                             typename common::data_traits<decltype(data)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, MakeFromContiguousAsStridedArray) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS_AS_STRIDED);

  auto data = common::make_data(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), data_pointer(data));
  EXPECT_EQ(1, data_nblocks(data));
  EXPECT_EQ(memory.num_blocks * memory.block_size, data_blocksize(data));
  EXPECT_EQ(0, data_stride(data));
  EXPECT_TRUE(data_iscontiguous(data));

  static_assert(common::is_data<decltype(data)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam, typename common::data_traits<decltype(data)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, MakeFromContiguousAsStridedArrayConst) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS_AS_STRIDED);

  auto data = common::make_data(const_cast<const TypeParam*>(memory.data.get()), memory.num_blocks,
                                memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), data_pointer(data));
  EXPECT_EQ(1, data_nblocks(data));
  EXPECT_EQ(memory.num_blocks * memory.block_size, data_blocksize(data));
  EXPECT_EQ(0, data_stride(data));
  EXPECT_TRUE(data_iscontiguous(data));

  EXPECT_TRUE(data_iscontiguous(data));

  static_assert(common::is_data<decltype(data)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<const TypeParam,
                             typename common::data_traits<decltype(data)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, MakeFromStridedArray) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_STRIDED);

  auto data = common::make_data(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), data_pointer(data));
  EXPECT_EQ(memory.num_blocks, data_nblocks(data));
  EXPECT_EQ(memory.block_size, data_blocksize(data));
  EXPECT_EQ(memory.stride, data_stride(data));
  EXPECT_FALSE(data_iscontiguous(data));

  static_assert(common::is_data<decltype(data)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam, typename common::data_traits<decltype(data)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, MakeFromStridedArrayConst) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_STRIDED);

  auto data = common::make_data(const_cast<const TypeParam*>(memory.data.get()), memory.num_blocks,
                                memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), data_pointer(data));
  EXPECT_EQ(memory.num_blocks, data_nblocks(data));
  EXPECT_EQ(memory.block_size, data_blocksize(data));
  EXPECT_EQ(memory.stride, data_stride(data));
  EXPECT_FALSE(data_iscontiguous(data));

  static_assert(common::is_data<decltype(data)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<const TypeParam,
                             typename common::data_traits<decltype(data)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, MakeBufferUniquePtr) {
  const std::size_t N = 13;
  auto data = common::Buffer<TypeParam>(N);

  EXPECT_NE(nullptr, data_pointer(data));
  EXPECT_EQ(1, data_nblocks(data));
  EXPECT_EQ(N, data_blocksize(data));
  EXPECT_EQ(0, data_stride(data));
  EXPECT_TRUE(data_iscontiguous(data));
}

TYPED_TEST(DataDescriptorTest, CtorFromPointer) {
  TypeParam value = 26;
  TypeParam* value_ptr = &value;

  auto data = common::DataDescriptor<TypeParam>{value_ptr, 1};

  EXPECT_EQ(value_ptr, data_pointer(data));
  EXPECT_EQ(1, data_nblocks(data));
  EXPECT_EQ(1, data_blocksize(data));
  EXPECT_EQ(0, data_stride(data));
  EXPECT_TRUE(data_iscontiguous(data));

  static_assert(common::is_data<decltype(data)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam, typename common::data_traits<decltype(data)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, CtorFromPointerConst) {
  TypeParam value = 26;
  const TypeParam* value_ptr = &value;

  auto data = common::DataDescriptor<const TypeParam>{value_ptr, 1};

  EXPECT_EQ(value_ptr, data_pointer(data));
  EXPECT_EQ(1, data_nblocks(data));
  EXPECT_EQ(1, data_blocksize(data));
  EXPECT_EQ(0, data_stride(data));
  EXPECT_TRUE(data_iscontiguous(data));

  static_assert(common::is_data<decltype(data)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<const TypeParam,
                             typename common::data_traits<decltype(data)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, CtorFromCArray) {
  const int N = 13;
  TypeParam value_array[N]{};

  auto data = common::DataDescriptor<decltype(value_array)>{value_array};

  EXPECT_EQ(&value_array[0], data_pointer(data));
  EXPECT_EQ(1, data_nblocks(data));
  EXPECT_EQ(std::extent<decltype(value_array)>::value, data_blocksize(data));
  EXPECT_EQ(0, data_stride(data));
  EXPECT_TRUE(data_iscontiguous(data));

  static_assert(common::is_data<decltype(data)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam, typename common::data_traits<decltype(data)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, CtorFromCArrayConst) {
  const int N = 13;
  const TypeParam value_array[N]{};

  auto data = common::DataDescriptor<decltype(value_array)>{value_array};

  EXPECT_EQ(&value_array[0], data_pointer(data));
  EXPECT_EQ(1, data_nblocks(data));
  EXPECT_EQ(std::extent<decltype(value_array)>::value, data_blocksize(data));
  EXPECT_EQ(0, data_stride(data));
  EXPECT_TRUE(data_iscontiguous(data));

  static_assert(common::is_data<decltype(data)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<const TypeParam,
                             typename common::data_traits<decltype(data)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, CtorFromContiguousArray) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS);

  auto data = common::make_data(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), data_pointer(data));
  EXPECT_EQ(memory.num_blocks, data_nblocks(data));
  EXPECT_EQ(memory.block_size, data_blocksize(data));
  EXPECT_EQ(memory.stride, data_stride(data));
  EXPECT_TRUE(data_iscontiguous(data));

  static_assert(common::is_data<decltype(data)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam, typename common::data_traits<decltype(data)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, CtorFromContiguousArrayConst) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS);

  auto data = common::make_data(const_cast<const TypeParam*>(memory.data.get()), memory.num_blocks,
                                memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), data_pointer(data));
  EXPECT_EQ(memory.num_blocks, data_nblocks(data));
  EXPECT_EQ(memory.block_size, data_blocksize(data));
  EXPECT_EQ(memory.stride, data_stride(data));
  EXPECT_TRUE(data_iscontiguous(data));

  static_assert(common::is_data<decltype(data)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<const TypeParam,
                             typename common::data_traits<decltype(data)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, CtorFromContiguousAsStridedArray) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS_AS_STRIDED);

  auto data = common::make_data(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), data_pointer(data));
  EXPECT_EQ(1, data_nblocks(data));
  EXPECT_EQ(memory.num_blocks * memory.block_size, data_blocksize(data));
  EXPECT_EQ(0, data_stride(data));
  EXPECT_TRUE(data_iscontiguous(data));

  static_assert(common::is_data<decltype(data)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam, typename common::data_traits<decltype(data)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, CtorFromContiguousAsStridedArrayConst) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS_AS_STRIDED);

  auto data = common::make_data(const_cast<const TypeParam*>(memory.data.get()), memory.num_blocks,
                                memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), data_pointer(data));
  EXPECT_EQ(1, data_nblocks(data));
  EXPECT_EQ(memory.num_blocks * memory.block_size, data_blocksize(data));
  EXPECT_EQ(0, data_stride(data));
  EXPECT_TRUE(data_iscontiguous(data));

  static_assert(common::is_data<decltype(data)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<const TypeParam,
                             typename common::data_traits<decltype(data)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, CtorFromStridedArray) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_STRIDED);

  auto data = common::make_data(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), data_pointer(data));
  EXPECT_EQ(memory.num_blocks, data_nblocks(data));
  EXPECT_EQ(memory.block_size, data_blocksize(data));
  EXPECT_EQ(memory.stride, data_stride(data));
  EXPECT_FALSE(data_iscontiguous(data));

  static_assert(common::is_data<decltype(data)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam, typename common::data_traits<decltype(data)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, CtorFromStridedArrayConst) {
  memory_data<TypeParam> memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_STRIDED);

  auto data = common::make_data(const_cast<const TypeParam*>(memory.data.get()), memory.num_blocks,
                                memory.block_size, memory.stride);

  EXPECT_EQ(memory.data.get(), data_pointer(data));
  EXPECT_EQ(memory.num_blocks, data_nblocks(data));
  EXPECT_EQ(memory.block_size, data_blocksize(data));
  EXPECT_EQ(memory.stride, data_stride(data));
  EXPECT_FALSE(data_iscontiguous(data));

  static_assert(common::is_data<decltype(data)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<const TypeParam,
                             typename common::data_traits<decltype(data)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, CtorBufferUniquePtr) {
  const std::size_t N = 13;
  auto data = common::Buffer<TypeParam>(N);

  EXPECT_NE(nullptr, data_pointer(data));
  EXPECT_EQ(1, data_nblocks(data));
  EXPECT_EQ(N, data_blocksize(data));
  EXPECT_EQ(0, data_stride(data));
  EXPECT_TRUE(data_iscontiguous(data));
}

template <class TypeParam, class Buffer>
void check_copy_ctor(Buffer& data) {
  auto data_copy = data;

  EXPECT_EQ(data_pointer(data), data_pointer(data_copy));
  EXPECT_EQ(data_nblocks(data), data_nblocks(data_copy));
  EXPECT_EQ(data_blocksize(data), data_blocksize(data_copy));
  EXPECT_EQ(data_stride(data), data_stride(data_copy));
  EXPECT_EQ(data_iscontiguous(data), data_iscontiguous(data_copy));

  static_assert(common::is_data<decltype(data_copy)>::value, "It should be a Buffer (concept)");

  static_assert(std::is_same<TypeParam,
                             typename common::data_traits<decltype(data_copy)>::element_t>::value,
                "Wrong type returned");
}

TYPED_TEST(DataDescriptorTest, CopyCtorFromPointer) {
  TypeParam value = 26;
  auto data = common::make_data(&value, 1);
  check_copy_ctor<TypeParam>(data);

  const TypeParam value_const = value;
  auto data_const = common::make_data(&value_const, 1);
  check_copy_ctor<const TypeParam>(data_const);
}

TYPED_TEST(DataDescriptorTest, CopyCtorFromCArray) {
  const int N = 13;
  TypeParam value_array[N]{};

  auto data = common::make_data(value_array, N);
  check_copy_ctor<TypeParam>(data);

  const TypeParam value_array_const[N]{};
  auto data_const = common::make_data(value_array_const, N);
  check_copy_ctor<const TypeParam>(data_const);
}

TYPED_TEST(DataDescriptorTest, CopyCtorFromContiguousArray) {
  auto memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS);

  auto data = common::make_data(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);
  check_copy_ctor<TypeParam>(data);

  auto data_const = common::make_data(static_cast<const TypeParam*>(memory.data.get()),
                                      memory.num_blocks, memory.block_size, memory.stride);
  check_copy_ctor<const TypeParam>(data_const);
}

TYPED_TEST(DataDescriptorTest, CopyCtorFromContiguousAsStridedArray) {
  auto memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS_AS_STRIDED);

  auto data = common::make_data(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);
  check_copy_ctor<TypeParam>(data);

  auto data_const = common::make_data(static_cast<const TypeParam*>(memory.data.get()),
                                      memory.num_blocks, memory.block_size, memory.stride);
  check_copy_ctor<const TypeParam>(data_const);
}

TYPED_TEST(DataDescriptorTest, CopyCtorFromStridedArray) {
  auto memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_STRIDED);

  auto data = common::make_data(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);
  check_copy_ctor<TypeParam>(data);

  auto data_const = common::make_data(const_cast<const TypeParam*>(memory.data.get()), memory.num_blocks,
                                      memory.block_size, memory.stride);
  check_copy_ctor<const TypeParam>(data_const);
}

template <class TypeParam, class Buffer>
void check_temporary(Buffer& data) {
  auto data_temp = create_temporary_buffer(data);

  EXPECT_NE(common::data_pointer(data), common::data_pointer(data_temp));
  EXPECT_EQ(common::data_count(data), common::data_count(data_temp));
  EXPECT_TRUE(common::data_iscontiguous(data_temp));
}

TYPED_TEST(DataDescriptorTest, CreateTemporaryFromPointer) {
  TypeParam value = 26;
  auto data = common::make_data(&value, 1);
  check_temporary<TypeParam>(data);

  const TypeParam value_const = value;
  auto data_const = common::make_data(&value_const, 1);
  check_temporary<const TypeParam>(data_const);
}

TYPED_TEST(DataDescriptorTest, CreateTemporaryFromCArray) {
  const int N = 13;
  TypeParam value_array[N]{};

  auto data = common::make_data(value_array, N);
  check_temporary<TypeParam>(data);

  const TypeParam value_array_const[N]{};
  auto data_const = common::make_data(value_array_const, N);
  check_temporary<const TypeParam>(data_const);
}

TYPED_TEST(DataDescriptorTest, CreateTemporaryFromContiguousArray) {
  auto memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS);

  auto data = common::make_data(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);
  check_temporary<TypeParam>(data);

  auto data_const = common::make_data(static_cast<const TypeParam*>(memory.data.get()),
                                      memory.num_blocks, memory.block_size, memory.stride);
  check_temporary<const TypeParam>(data_const);
}

TYPED_TEST(DataDescriptorTest, CreateTemporaryFromContiguousAsStridedArray) {
  auto memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_CONTIGUOUS_AS_STRIDED);

  auto data = common::make_data(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);
  check_temporary<TypeParam>(data);

  auto data_const = common::make_data(static_cast<const TypeParam*>(memory.data.get()),
                                      memory.num_blocks, memory.block_size, memory.stride);
  check_temporary<const TypeParam>(data_const);
}

TYPED_TEST(DataDescriptorTest, CreateTemporaryFromStridedArray) {
  auto memory = create_memory<TypeParam>(MEMORY_TYPE::ARRAY_STRIDED);

  auto data = common::make_data(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);
  check_temporary<TypeParam>(data);

  auto data_const = common::make_data(const_cast<const TypeParam*>(memory.data.get()), memory.num_blocks,
                                      memory.block_size, memory.stride);
  check_temporary<const TypeParam>(data_const);
}

template <class T>
auto create_data_from_memory(memory_data<T>& memory) {
  return common::make_data(memory.data.get(), memory.num_blocks, memory.block_size, memory.stride);
}

template <class T>
auto create_const_data_from_memory(memory_data<T>& memory) {
  return common::make_data(const_cast<const T*>(memory.data.get()), memory.num_blocks, memory.block_size,
                           memory.stride);
}

TYPED_TEST(DataDescriptorTest, CopyDataCArrays) {
  const int N = 13;
  TypeParam memory_src[N];
  TypeParam memory_dst[N];

  for (int i = 0; i < N; ++i)
    memory_src[i] = dlaf_test::TypeUtilities<TypeParam>::element(i, 0);

  auto data_src = common::make_data(const_cast<const TypeParam*>(memory_src), N);
  auto data_dest = common::make_data(memory_dst, N);

  common::copy(data_src, data_dest);

  for (int i = 0; i < N; ++i)
    EXPECT_EQ(dlaf_test::TypeUtilities<TypeParam>::element(i, 0), memory_dst[i]);
}

TYPED_TEST(DataDescriptorTest, CopyDataArrays) {
  auto memory_types = {MEMORY_TYPE::ARRAY_CONTIGUOUS, MEMORY_TYPE::ARRAY_CONTIGUOUS_AS_STRIDED,
                       MEMORY_TYPE::ARRAY_STRIDED};

  for (auto memory_type : memory_types) {
    auto memory_src = create_memory<TypeParam>(memory_type);
    auto memory_dest = create_memory<TypeParam>(memory_type);

    for (std::size_t i = 0; i < memory_src.num_blocks * memory_src.block_size; ++i)
      memory_src[i] = dlaf_test::TypeUtilities<TypeParam>::element(i, 0);

    auto data_src = create_const_data_from_memory(memory_src);
    auto data_dest = create_data_from_memory(memory_dest);

    common::copy(data_src, data_dest);

    for (std::size_t i = 0; i < memory_src.num_blocks * memory_src.block_size; ++i)
      EXPECT_EQ(dlaf_test::TypeUtilities<TypeParam>::element(i, 0), memory_dest[i]);
  }
}

TYPED_TEST(DataDescriptorTest, CopyDataHeterogeneous) {
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

    auto data_array = common::make_data(memory_array, N);
    auto data_dest = create_data_from_memory(memory_dest);

    copy(data_array, data_dest);

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

    auto data_src = create_const_data_from_memory(memory_src);
    auto data_array = common::make_data(memory_array, N);

    copy(data_src, data_array);

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

      auto data_src = create_const_data_from_memory(memory_src);
      auto data_dest = create_data_from_memory(memory_dest);

      copy(data_src, data_dest);

      for (std::size_t i = 0; i < N; ++i)
        EXPECT_EQ(dlaf_test::TypeUtilities<TypeParam>::element(i, 0), memory_dest[i]);
    }
  }
}
