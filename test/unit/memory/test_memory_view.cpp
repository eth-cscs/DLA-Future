//
// NS3C
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "ns3c/memory/memory_view.h"

#include "gtest/gtest.h"
#include "ns3c_test/util_types.h"

using namespace ns3c;
using namespace ns3c_test;
using namespace testing;

template <typename Type>
class MemoryViewTest : public ::testing::Test {};

TYPED_TEST_CASE(MemoryViewTest, ElementTypes);

int size = 397;

TYPED_TEST(MemoryViewTest, ConstructorAllocates) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> mem(size);

  EXPECT_EQ(size, mem.size());
  EXPECT_NE(nullptr, mem());
  Type* ptr = mem();
  for (std::size_t i = 0; i < mem.size(); ++i)
    EXPECT_EQ(ptr + i, mem(i));
}

TYPED_TEST(MemoryViewTest, ConstructorPointer) {
  using Type = TypeParam;
  Type* ptr = new Type[size];

  memory::MemoryView<Type, Device::CPU> mem(ptr, size);

  EXPECT_EQ(size, mem.size());
  EXPECT_NE(nullptr, mem());

  for (std::size_t i = 0; i < mem.size(); ++i)
    EXPECT_EQ(ptr + i, mem(i));

  free(ptr);
}

TYPED_TEST(MemoryViewTest, ConstructorSubview) {
  using Type = TypeParam;
  Type* ptr = new Type[size];

  memory::MemoryView<Type, Device::CPU> mem(ptr, size);
  memory::MemoryView<Type, Device::CPU> mem2(mem, 4, size - 5);

  EXPECT_EQ(size - 5, mem2.size());
  EXPECT_NE(nullptr, mem2());

  for (std::size_t i = 0; i < mem2.size(); ++i)
    EXPECT_EQ(ptr + 4 + i, mem2(i));

  memory::MemoryView<Type, Device::CPU> mem3(mem2, 4, size - 15);

  EXPECT_EQ(size - 15, mem3.size());
  EXPECT_NE(nullptr, mem3());

  for (std::size_t i = 0; i < mem3.size(); ++i)
    EXPECT_EQ(ptr + 8 + i, mem3(i));

  free(ptr);
}

TYPED_TEST(MemoryViewTest, CopyConstructor) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> mem0(size + 2);
  memory::MemoryView<Type, Device::CPU> mem(mem0, 1, size);

  memory::MemoryView<Type, Device::CPU> mem2(mem);

  EXPECT_EQ(mem(), mem2());
  EXPECT_EQ(mem.size(), mem2.size());

  Type* ptr = mem();
  for (std::size_t i = 0; i < mem2.size(); ++i)
    EXPECT_EQ(ptr + i, mem2(i));
}

TYPED_TEST(MemoryViewTest, MoveConstructor) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> mem0(size + 2);
  memory::MemoryView<Type, Device::CPU> mem(mem0, 1, size);
  Type* ptr = mem();

  memory::MemoryView<Type, Device::CPU> mem2(std::move(mem));

  EXPECT_EQ(size, mem2.size());
  for (std::size_t i = 0; i < mem2.size(); ++i)
    EXPECT_EQ(ptr + i, mem2(i));
}

TYPED_TEST(MemoryViewTest, CopyAssignement) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> mem0(size + 2);
  memory::MemoryView<Type, Device::CPU> mem(mem0, 1, size);
  memory::MemoryView<Type, Device::CPU> mem2(size - 5);
  EXPECT_NE(mem(), mem2());

  mem2 = mem;

  EXPECT_EQ(mem(), mem2());
  EXPECT_EQ(mem.size(), mem2.size());

  Type* ptr = mem();
  for (std::size_t i = 0; i < mem2.size(); ++i)
    EXPECT_EQ(ptr + i, mem2(i));
}

TYPED_TEST(MemoryViewTest, MoveAssignement) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> mem0(size + 2);
  memory::MemoryView<Type, Device::CPU> mem(mem0, 1, size);
  memory::MemoryView<Type, Device::CPU> mem2(size - 5);
  EXPECT_NE(mem(), mem2());
  Type* ptr = mem();

  mem2 = std::move(mem);

  EXPECT_EQ(size, mem2.size());
  for (std::size_t i = 0; i < mem2.size(); ++i)
    EXPECT_EQ(ptr + i, mem2(i));
}
