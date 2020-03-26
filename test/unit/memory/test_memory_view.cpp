//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/memory/memory_view.h"

#include "gtest/gtest.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf_test;
using namespace testing;

template <typename Type>
class MemoryViewTest : public ::testing::Test {};

TYPED_TEST_SUITE(MemoryViewTest, ElementTypes);

constexpr std::size_t size = 397;

TYPED_TEST(MemoryViewTest, ConstructorAllocates) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> mem(size);

  EXPECT_EQ(size, mem.size());
  EXPECT_NE(nullptr, mem());
  Type* ptr = mem();
  for (std::size_t i = 0; i < mem.size(); ++i)
    EXPECT_EQ(ptr + i, mem(i));

  memory::MemoryView<Type, Device::CPU> mem2(0);
  EXPECT_EQ(0, mem2.size());
  EXPECT_EQ(nullptr, mem2());
}

TYPED_TEST(MemoryViewTest, ConstructorPointer) {
  using Type = TypeParam;
  std::unique_ptr<Type[]> uptr(new Type[size]);
  Type* ptr = uptr.get();

  memory::MemoryView<Type, Device::CPU> mem(ptr, size);

  EXPECT_EQ(size, mem.size());
  EXPECT_EQ(ptr, mem());

  for (std::size_t i = 0; i < mem.size(); ++i)
    EXPECT_EQ(ptr + i, mem(i));

  memory::MemoryView<Type, Device::CPU> mem2(ptr, 0);
  EXPECT_EQ(0, mem2.size());
  EXPECT_EQ(nullptr, mem2());

  memory::MemoryView<Type, Device::CPU> mem3(nullptr, 0);
  EXPECT_EQ(0, mem3.size());
  EXPECT_EQ(nullptr, mem3());
}

TYPED_TEST(MemoryViewTest, ConstructorPointerConst) {
  using Type = TypeParam;
  std::unique_ptr<Type[]> uptr(new Type[size]);
  const Type* const_ptr = uptr.get();
  memory::MemoryView<const Type, Device::CPU> const_mem(const_ptr, size);

  EXPECT_EQ(size, const_mem.size());
  EXPECT_EQ(const_ptr, const_mem());

  for (std::size_t i = 0; i < const_mem.size(); ++i)
    EXPECT_EQ(const_ptr + i, const_mem(i));

  memory::MemoryView<const Type, Device::CPU> const_mem2(const_ptr, 0);
  EXPECT_EQ(0, const_mem2.size());
  EXPECT_EQ(nullptr, const_mem2());

  memory::MemoryView<const Type, Device::CPU> const_mem3(nullptr, 0);
  EXPECT_EQ(0, const_mem3.size());
  EXPECT_EQ(nullptr, const_mem3());
}

TYPED_TEST(MemoryViewTest, ConstructorSubview) {
  using Type = TypeParam;
  std::unique_ptr<Type[]> uptr(new Type[size]);
  Type* ptr = uptr.get();

  memory::MemoryView<Type, Device::CPU> mem(ptr, size);
  memory::MemoryView<Type, Device::CPU> mem2(mem, 4, size - 5);

  EXPECT_EQ(size - 5, mem2.size());
  EXPECT_EQ(ptr + 4, mem2());

  for (std::size_t i = 0; i < mem2.size(); ++i)
    EXPECT_EQ(ptr + 4 + i, mem2(i));

  memory::MemoryView<Type, Device::CPU> mem3(mem2, 4, size - 15);

  EXPECT_EQ(size - 15, mem3.size());
  EXPECT_EQ(ptr + 8, mem3());

  for (std::size_t i = 0; i < mem3.size(); ++i)
    EXPECT_EQ(ptr + 8 + i, mem3(i));

  memory::MemoryView<Type, Device::CPU> mem4(mem, 4, 0);

  EXPECT_EQ(0, mem4.size());
  EXPECT_EQ(nullptr, mem4());
}

TYPED_TEST(MemoryViewTest, ConstructorSubviewConst) {
  using Type = TypeParam;
  std::unique_ptr<Type[]> uptr(new Type[size]);
  Type* ptr = uptr.get();

  memory::MemoryView<const Type, Device::CPU> const_mem(ptr, size);
  memory::MemoryView<const Type, Device::CPU> const_mem2(const_mem, 4, size - 5);

  EXPECT_EQ(size - 5, const_mem2.size());
  EXPECT_EQ(ptr + 4, const_mem2());

  for (std::size_t i = 0; i < const_mem2.size(); ++i)
    EXPECT_EQ(ptr + 4 + i, const_mem2(i));

  memory::MemoryView<const Type, Device::CPU> const_mem3(const_mem2, 4, size - 15);

  EXPECT_EQ(size - 15, const_mem3.size());
  EXPECT_EQ(ptr + 8, const_mem3());

  for (std::size_t i = 0; i < const_mem3.size(); ++i)
    EXPECT_EQ(ptr + 8 + i, const_mem3(i));

  memory::MemoryView<const Type, Device::CPU> const_mem4(const_mem, 4, 0);

  EXPECT_EQ(0, const_mem4.size());
  EXPECT_EQ(nullptr, const_mem4());
}

TYPED_TEST(MemoryViewTest, ConstructorSubviewMix) {
  using Type = TypeParam;
  std::unique_ptr<Type[]> uptr(new Type[size]);
  Type* ptr = uptr.get();

  memory::MemoryView<Type, Device::CPU> mem(ptr, size);
  memory::MemoryView<const Type, Device::CPU> const_mem2(mem, 4, size - 5);

  EXPECT_EQ(size - 5, const_mem2.size());
  EXPECT_EQ(ptr + 4, const_mem2());

  for (std::size_t i = 0; i < const_mem2.size(); ++i)
    EXPECT_EQ(ptr + 4 + i, const_mem2(i));

  memory::MemoryView<Type, Device::CPU> const_mem4(mem, 4, 0);

  EXPECT_EQ(0, const_mem4.size());
  EXPECT_EQ(nullptr, const_mem4());
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

TYPED_TEST(MemoryViewTest, CopyConstructorConst) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> mem0(size + 2);
  memory::MemoryView<const Type, Device::CPU> const_mem(mem0, 1, size);

  memory::MemoryView<const Type, Device::CPU> const_mem2(const_mem);

  EXPECT_EQ(const_mem(), const_mem2());
  EXPECT_EQ(const_mem.size(), const_mem2.size());

  const Type* ptr = const_mem();
  for (std::size_t i = 0; i < const_mem2.size(); ++i)
    EXPECT_EQ(ptr + i, const_mem2(i));
}

TYPED_TEST(MemoryViewTest, CopyConstructorMix) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> mem0(size + 2);
  memory::MemoryView<Type, Device::CPU> mem(mem0, 1, size);

  memory::MemoryView<const Type, Device::CPU> const_mem2(mem);

  EXPECT_EQ(mem(), const_mem2());
  EXPECT_EQ(mem.size(), const_mem2.size());

  Type* ptr = mem();
  for (std::size_t i = 0; i < const_mem2.size(); ++i)
    EXPECT_EQ(ptr + i, const_mem2(i));
}

TYPED_TEST(MemoryViewTest, MoveConstructor) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> mem0(size + 2);
  memory::MemoryView<Type, Device::CPU> mem(mem0, 1, size);
  Type* ptr = mem();

  memory::MemoryView<Type, Device::CPU> mem2(std::move(mem));

  EXPECT_EQ(nullptr, mem());
  EXPECT_EQ(0, mem.size());

  EXPECT_EQ(size, mem2.size());
  for (std::size_t i = 0; i < mem2.size(); ++i)
    EXPECT_EQ(ptr + i, mem2(i));
}

TYPED_TEST(MemoryViewTest, MoveConstructorConst) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> mem0(size + 2);
  memory::MemoryView<const Type, Device::CPU> const_mem(mem0, 1, size);
  const Type* ptr = const_mem();

  memory::MemoryView<const Type, Device::CPU> const_mem2(std::move(const_mem));

  EXPECT_EQ(nullptr, const_mem());
  EXPECT_EQ(0, const_mem.size());

  EXPECT_EQ(size, const_mem2.size());
  for (std::size_t i = 0; i < const_mem2.size(); ++i)
    EXPECT_EQ(ptr + i, const_mem2(i));
}

TYPED_TEST(MemoryViewTest, MoveConstructorMix) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> mem0(size + 2);
  memory::MemoryView<Type, Device::CPU> mem(mem0, 1, size);
  Type* ptr = mem();

  memory::MemoryView<const Type, Device::CPU> const_mem2(std::move(mem));

  EXPECT_EQ(nullptr, mem());
  EXPECT_EQ(0, mem.size());

  EXPECT_EQ(size, const_mem2.size());
  for (std::size_t i = 0; i < const_mem2.size(); ++i)
    EXPECT_EQ(ptr + i, const_mem2(i));
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

TYPED_TEST(MemoryViewTest, CopyAssignementConst) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> mem0(size + 2);
  memory::MemoryView<const Type, Device::CPU> const_mem(mem0, 1, size);
  memory::MemoryView<const Type, Device::CPU> const_mem2;
  EXPECT_NE(const_mem(), const_mem2());

  const_mem2 = const_mem;

  EXPECT_EQ(const_mem(), const_mem2());
  EXPECT_EQ(const_mem.size(), const_mem2.size());

  const Type* ptr = const_mem();
  for (std::size_t i = 0; i < const_mem2.size(); ++i)
    EXPECT_EQ(ptr + i, const_mem2(i));
}

TYPED_TEST(MemoryViewTest, CopyAssignementMix) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> mem0(size + 2);
  memory::MemoryView<Type, Device::CPU> mem(mem0, 1, size);
  memory::MemoryView<const Type, Device::CPU> const_mem2;
  EXPECT_NE(mem(), const_mem2());

  const_mem2 = mem;

  EXPECT_EQ(mem(), const_mem2());
  EXPECT_EQ(mem.size(), const_mem2.size());

  Type* ptr = mem();
  for (std::size_t i = 0; i < const_mem2.size(); ++i)
    EXPECT_EQ(ptr + i, const_mem2(i));
}

TYPED_TEST(MemoryViewTest, MoveAssignement) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> mem0(size + 2);
  memory::MemoryView<Type, Device::CPU> mem(mem0, 1, size);
  memory::MemoryView<Type, Device::CPU> mem2(size - 5);
  EXPECT_NE(mem(), mem2());
  Type* ptr = mem();

  mem2 = std::move(mem);

  EXPECT_EQ(nullptr, mem());
  EXPECT_EQ(0, mem.size());

  EXPECT_EQ(size, mem2.size());
  for (std::size_t i = 0; i < mem2.size(); ++i)
    EXPECT_EQ(ptr + i, mem2(i));
}

TYPED_TEST(MemoryViewTest, MoveAssignementConst) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> mem0(size + 2);
  memory::MemoryView<const Type, Device::CPU> const_mem(mem0, 1, size);
  memory::MemoryView<const Type, Device::CPU> const_mem2;
  EXPECT_NE(const_mem(), const_mem2());
  const Type* ptr = const_mem();

  const_mem2 = std::move(const_mem);

  EXPECT_EQ(nullptr, const_mem());
  EXPECT_EQ(0, const_mem.size());

  EXPECT_EQ(size, const_mem2.size());
  for (std::size_t i = 0; i < const_mem2.size(); ++i)
    EXPECT_EQ(ptr + i, const_mem2(i));
}

TYPED_TEST(MemoryViewTest, MoveAssignementMix) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> mem0(size + 2);
  memory::MemoryView<Type, Device::CPU> mem(mem0, 1, size);
  memory::MemoryView<const Type, Device::CPU> const_mem2;
  EXPECT_NE(mem(), const_mem2());
  Type* ptr = mem();

  const_mem2 = std::move(mem);

  EXPECT_EQ(nullptr, mem());
  EXPECT_EQ(0, mem.size());

  EXPECT_EQ(size, const_mem2.size());
  for (std::size_t i = 0; i < const_mem2.size(); ++i)
    EXPECT_EQ(ptr + i, const_mem2(i));
}
