//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <memory>
#include <utility>

#include <dlaf/memory/memory_chunk.h>

#include <gtest/gtest.h>

#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::test;
using namespace testing;

template <typename Type>
class MemoryChunkTest : public ::testing::Test {};

TYPED_TEST_SUITE(MemoryChunkTest, ElementTypes);

constexpr SizeType size = 397;

std::vector<memory::AllocateOn> allocate_ons = {memory::AllocateOn::Construction,
                                                memory::AllocateOn::Demand};

TYPED_TEST(MemoryChunkTest, ConstructorAllocates) {
  using Type = TypeParam;
  for (const auto& allocate_on : allocate_ons) {
    memory::MemoryChunk<Type, Device::CPU> mem(size, allocate_on);

    EXPECT_EQ(size, mem.size());
    EXPECT_NE(nullptr, mem());
    Type* ptr = mem();
    for (SizeType i = 0; i < mem.size(); ++i)
      EXPECT_EQ(ptr + i, mem(i));
  }
}

TYPED_TEST(MemoryChunkTest, ConstructorPointer) {
  using Type = TypeParam;
  std::unique_ptr<Type[]> uptr(new Type[size]);
  Type* ptr = uptr.get();

  memory::MemoryChunk<Type, Device::CPU> mem(ptr, size);

  EXPECT_EQ(size, mem.size());
  EXPECT_NE(nullptr, mem());

  for (SizeType i = 0; i < mem.size(); ++i)
    EXPECT_EQ(ptr + i, mem(i));
}

TYPED_TEST(MemoryChunkTest, MoveConstructor) {
  using Type = TypeParam;
  for (const auto& allocate_on : allocate_ons) {
    memory::MemoryChunk<Type, Device::CPU> mem(size, allocate_on);
    Type* ptr = mem();

    memory::MemoryChunk<Type, Device::CPU> mem2(std::move(mem));

    EXPECT_EQ(nullptr, mem());
    EXPECT_EQ(0, mem.size());

    EXPECT_EQ(size, mem2.size());
    for (SizeType i = 0; i < mem2.size(); ++i)
      EXPECT_EQ(ptr + i, mem2(i));
  }
}

TYPED_TEST(MemoryChunkTest, MoveAssignment) {
  using Type = TypeParam;
  for (const auto& allocate_on : allocate_ons) {
    memory::MemoryChunk<Type, Device::CPU> mem(size, allocate_on);
    memory::MemoryChunk<Type, Device::CPU> mem2(size - 5, allocate_on);
    EXPECT_NE(mem(), mem2());
    Type* ptr = mem();

    mem2 = std::move(mem);

    EXPECT_EQ(nullptr, mem());
    EXPECT_EQ(0, mem.size());

    EXPECT_EQ(size, mem2.size());
    for (SizeType i = 0; i < mem2.size(); ++i)
      EXPECT_EQ(ptr + i, mem2(i));
  }
}
