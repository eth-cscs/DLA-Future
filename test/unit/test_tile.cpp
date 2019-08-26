//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "ns3c/tile.h"

#include <stdexcept>

#include "gtest/gtest.h"
#include "ns3c/memory/memory_view.h"
#include "ns3c_test/util_types.h"

using namespace ns3c;
using namespace ns3c_test;
using namespace testing;

int m = 37;
int n = 87;
int ld = 133;

template <typename Type>
class TileTest : public ::testing::Test {};

TYPED_TEST_CASE(TileTest, MatrixElementTypes);

TYPED_TEST(TileTest, Constructor) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  Tile<Type, Device::CPU> tile(m, n, memory_view, ld);

  EXPECT_EQ(m, tile.m());
  EXPECT_EQ(n, tile.n());
  EXPECT_EQ(ld, tile.ld());

  for (int j = 0; j < tile.n(); ++j)
    for (int i = 0; i < tile.m(); ++i) {
      Type el = TypeUtilities<Type>::element(i + 0.01 * j, j - 0.01 * i);
      tile(i, j) = el;
      EXPECT_EQ(el, tile(i, j));
      EXPECT_EQ(el, *memory_view(i + ld * j));
      EXPECT_EQ(memory_view(i + ld * j), tile.ptr(i, j));
    }
}

TYPED_TEST(TileTest, ConstructorConst) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);
  memory::MemoryView<const Type, Device::CPU> c_memory_view(memory_view);

  Tile<const Type, Device::CPU> tile(m, n, c_memory_view, ld);

  EXPECT_EQ(m, tile.m());
  EXPECT_EQ(n, tile.n());
  EXPECT_EQ(ld, tile.ld());

  for (int j = 0; j < tile.n(); ++j)
    for (int i = 0; i < tile.m(); ++i) {
      EXPECT_EQ(memory_view(i + ld * j), tile.ptr(i, j));
    }
}

TYPED_TEST(TileTest, ConstructorMix) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  Tile<const Type, Device::CPU> tile(m, n, memory_view, ld);

  EXPECT_EQ(m, tile.m());
  EXPECT_EQ(n, tile.n());
  EXPECT_EQ(ld, tile.ld());

  for (int j = 0; j < tile.n(); ++j)
    for (int i = 0; i < tile.m(); ++i) {
      Type el = TypeUtilities<Type>::element(i + 0.01 * j, j - 0.01 * i);
      *memory_view(i + ld * j) = el;
      EXPECT_EQ(el, tile(i, j));
      EXPECT_EQ(el, *memory_view(i + ld * j));
      EXPECT_EQ(memory_view(i + ld * j), tile.ptr(i, j));
    }
}

TYPED_TEST(TileTest, ConstructorExceptions) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * (n - 1) + m - 1);
  memory::MemoryView<const Type, Device::CPU> c_memory_view(memory_view);

  EXPECT_THROW((Tile<Type, Device::CPU>(m, n, memory_view, ld)), std::invalid_argument);
  EXPECT_THROW((Tile<Type, Device::CPU>(-1, n, memory_view, ld)), std::invalid_argument);
  EXPECT_THROW((Tile<Type, Device::CPU>(m, -1, memory_view, ld)), std::invalid_argument);
  EXPECT_THROW((Tile<Type, Device::CPU>(m, n, memory_view, m - 1)), std::invalid_argument);
  EXPECT_THROW((Tile<Type, Device::CPU>(0, n, memory_view, 0)), std::invalid_argument);

  EXPECT_THROW((Tile<const Type, Device::CPU>(m, n, c_memory_view, ld)), std::invalid_argument);
  EXPECT_THROW((Tile<const Type, Device::CPU>(-1, n, c_memory_view, ld)), std::invalid_argument);
  EXPECT_THROW((Tile<const Type, Device::CPU>(m, -1, c_memory_view, ld)), std::invalid_argument);
  EXPECT_THROW((Tile<const Type, Device::CPU>(m, n, c_memory_view, m - 1)), std::invalid_argument);
  EXPECT_THROW((Tile<const Type, Device::CPU>(0, n, c_memory_view, 0)), std::invalid_argument);

  EXPECT_THROW((Tile<const Type, Device::CPU>(m, n, memory_view, ld)), std::invalid_argument);
  EXPECT_THROW((Tile<const Type, Device::CPU>(-1, n, memory_view, ld)), std::invalid_argument);
  EXPECT_THROW((Tile<const Type, Device::CPU>(m, -1, memory_view, ld)), std::invalid_argument);
  EXPECT_THROW((Tile<const Type, Device::CPU>(m, n, memory_view, m - 1)), std::invalid_argument);
  EXPECT_THROW((Tile<const Type, Device::CPU>(0, n, memory_view, 0)), std::invalid_argument);
}

TYPED_TEST(TileTest, MoveConstructor) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  Tile<Type, Device::CPU> tile0(m, n, memory_view, ld);

  Tile<Type, Device::CPU> tile(std::move(tile0));
  EXPECT_EQ(0, tile0.m());
  EXPECT_EQ(0, tile0.n());
  EXPECT_EQ(1, tile0.ld());

  EXPECT_EQ(m, tile.m());
  EXPECT_EQ(n, tile.n());
  EXPECT_EQ(ld, tile.ld());

  for (int j = 0; j < tile.n(); ++j)
    for (int i = 0; i < tile.m(); ++i) {
      EXPECT_EQ(tile.ptr(i, j), memory_view(i + ld * j));
    }
}

TYPED_TEST(TileTest, MoveConstructorConst) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  Tile<const Type, Device::CPU> const_tile0(m, n, memory_view, ld);

  Tile<const Type, Device::CPU> const_tile(std::move(const_tile0));
  EXPECT_EQ(0, const_tile0.m());
  EXPECT_EQ(0, const_tile0.n());
  EXPECT_EQ(1, const_tile0.ld());

  EXPECT_EQ(m, const_tile.m());
  EXPECT_EQ(n, const_tile.n());
  EXPECT_EQ(ld, const_tile.ld());

  for (int j = 0; j < const_tile.n(); ++j)
    for (int i = 0; i < const_tile.m(); ++i) {
      EXPECT_EQ(const_tile.ptr(i, j), memory_view(i + ld * j));
    }
}

TYPED_TEST(TileTest, MoveConstructorMix) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  Tile<Type, Device::CPU> tile0(m, n, memory_view, ld);

  Tile<const Type, Device::CPU> const_tile(std::move(tile0));
  EXPECT_EQ(0, tile0.m());
  EXPECT_EQ(0, tile0.n());
  EXPECT_EQ(1, tile0.ld());

  EXPECT_EQ(m, const_tile.m());
  EXPECT_EQ(n, const_tile.n());
  EXPECT_EQ(ld, const_tile.ld());

  for (int j = 0; j < const_tile.n(); ++j)
    for (int i = 0; i < const_tile.m(); ++i) {
      EXPECT_EQ(const_tile.ptr(i, j), memory_view(i + ld * j));
    }
}

TYPED_TEST(TileTest, MoveAssignement) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  Tile<Type, Device::CPU> tile0(m, n, memory_view, ld);
  Tile<Type, Device::CPU> tile(1, 1, memory::MemoryView<Type, Device::CPU>(1), 1);

  tile = std::move(tile0);
  EXPECT_EQ(0, tile0.m());
  EXPECT_EQ(0, tile0.n());
  EXPECT_EQ(1, tile0.ld());

  EXPECT_EQ(m, tile.m());
  EXPECT_EQ(n, tile.n());
  EXPECT_EQ(ld, tile.ld());

  for (int j = 0; j < tile.n(); ++j)
    for (int i = 0; i < tile.m(); ++i) {
      EXPECT_EQ(tile.ptr(i, j), memory_view(i + ld * j));
    }
}

TYPED_TEST(TileTest, MoveAssignementConst) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  Tile<const Type, Device::CPU> const_tile0(m, n, memory_view, ld);
  Tile<const Type, Device::CPU> const_tile(1, 1, memory::MemoryView<Type, Device::CPU>(1), 1);

  const_tile = std::move(const_tile0);
  EXPECT_EQ(0, const_tile0.m());
  EXPECT_EQ(0, const_tile0.n());
  EXPECT_EQ(1, const_tile0.ld());

  EXPECT_EQ(m, const_tile.m());
  EXPECT_EQ(n, const_tile.n());
  EXPECT_EQ(ld, const_tile.ld());

  for (int j = 0; j < const_tile.n(); ++j)
    for (int i = 0; i < const_tile.m(); ++i) {
      EXPECT_EQ(const_tile.ptr(i, j), memory_view(i + ld * j));
    }
}

TYPED_TEST(TileTest, MoveAssignementMix) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  Tile<Type, Device::CPU> tile0(m, n, memory_view, ld);
  Tile<const Type, Device::CPU> const_tile(1, 1, memory::MemoryView<Type, Device::CPU>(1), 1);

  const_tile = std::move(tile0);
  EXPECT_EQ(0, tile0.m());
  EXPECT_EQ(0, tile0.n());
  EXPECT_EQ(1, tile0.ld());

  EXPECT_EQ(m, const_tile.m());
  EXPECT_EQ(n, const_tile.n());
  EXPECT_EQ(ld, const_tile.ld());

  for (int j = 0; j < const_tile.n(); ++j)
    for (int i = 0; i < const_tile.m(); ++i) {
      EXPECT_EQ(const_tile.ptr(i, j), memory_view(i + ld * j));
    }
}
