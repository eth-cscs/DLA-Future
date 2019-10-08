//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/tile.h"

#include <stdexcept>
#include "gtest/gtest.h"
#include "dlaf/matrix/index.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf_test;
using namespace testing;

SizeType m = 37;
SizeType n = 87;
SizeType ld = 133;

std::size_t elIndex(SizeType i, SizeType j, SizeType ld) {
  using util::size_t::sum;
  using util::size_t::mul;
  return sum(i, mul(ld, j));
}

using TileSizes = std::tuple<TileElementSize, SizeType>;

template <class T, Device device>
TileSizes getSizes(const Tile<T, device>& tile) {
  return TileSizes(tile.size(), tile.ld());
}

template <typename Type>
class TileTest : public ::testing::Test {};

TYPED_TEST_CASE(TileTest, MatrixElementTypes);

TYPED_TEST(TileTest, Constructor) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  TileElementSize size(m, n);
  Tile<Type, Device::CPU> tile(size, memory_view, ld);
  EXPECT_EQ(TileSizes(size, ld), getSizes(tile));

  for (SizeType j = 0; j < tile.size().cols(); ++j)
    for (SizeType i = 0; i < tile.size().rows(); ++i) {
      Type el = TypeUtilities<Type>::element(i + 0.01 * j, j - 0.01 * i);
      tile(TileElementIndex(i, j)) = el;
      EXPECT_EQ(el, tile(TileElementIndex(i, j)));
      EXPECT_EQ(el, *memory_view(elIndex(i, j, ld)));
      EXPECT_EQ(memory_view(elIndex(i, j, ld)), tile.ptr(TileElementIndex(i, j)));
    }
}

TYPED_TEST(TileTest, ConstructorConst) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  TileElementSize size(m, n);
  Tile<const Type, Device::CPU> tile(size, memory_view, ld);
  EXPECT_EQ(TileSizes(size, ld), getSizes(tile));

  for (SizeType j = 0; j < tile.size().cols(); ++j)
    for (SizeType i = 0; i < tile.size().rows(); ++i) {
      EXPECT_EQ(memory_view(elIndex(i, j, ld)), tile.ptr(TileElementIndex(i, j)));
    }
}

TYPED_TEST(TileTest, ConstructorMix) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  TileElementSize size(m, n);
  Tile<const Type, Device::CPU> tile(size, memory_view, ld);
  EXPECT_EQ(TileSizes(size, ld), getSizes(tile));

  for (SizeType j = 0; j < tile.size().cols(); ++j)
    for (SizeType i = 0; i < tile.size().rows(); ++i) {
      Type el = TypeUtilities<Type>::element(i + 0.01 * j, j - 0.01 * i);
      *memory_view(elIndex(i, j, ld)) = el;
      EXPECT_EQ(el, tile(TileElementIndex(i, j)));
      EXPECT_EQ(el, *memory_view(elIndex(i, j, ld)));
      EXPECT_EQ(memory_view(elIndex(i, j, ld)), tile.ptr(TileElementIndex(i, j)));
    }
}

TYPED_TEST(TileTest, ConstructorExceptions) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * (n - 1) + m - 1);

  EXPECT_THROW((Tile<Type, Device::CPU>({m, n}, memory_view, ld)), std::invalid_argument);
  EXPECT_THROW((Tile<Type, Device::CPU>({-1, n}, memory_view, ld)), std::invalid_argument);
  EXPECT_THROW((Tile<Type, Device::CPU>({m, -1}, memory_view, ld)), std::invalid_argument);
  EXPECT_THROW((Tile<Type, Device::CPU>({m, n}, memory_view, m - 1)), std::invalid_argument);
  EXPECT_THROW((Tile<Type, Device::CPU>({0, n}, memory_view, 0)), std::invalid_argument);

  EXPECT_THROW((Tile<const Type, Device::CPU>({m, n}, memory_view, ld)), std::invalid_argument);
  EXPECT_THROW((Tile<const Type, Device::CPU>({-1, n}, memory_view, ld)), std::invalid_argument);
  EXPECT_THROW((Tile<const Type, Device::CPU>({m, -1}, memory_view, ld)), std::invalid_argument);
  EXPECT_THROW((Tile<const Type, Device::CPU>({m, n}, memory_view, m - 1)), std::invalid_argument);
  EXPECT_THROW((Tile<const Type, Device::CPU>({0, n}, memory_view, 0)), std::invalid_argument);
}

TYPED_TEST(TileTest, MoveConstructor) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  TileElementSize size(m, n);
  Tile<Type, Device::CPU> tile0(size, memory_view, ld);

  Tile<Type, Device::CPU> tile(std::move(tile0));
  EXPECT_EQ(TileSizes({0, 0}, 1), getSizes(tile0));
  EXPECT_EQ(TileSizes(size, ld), getSizes(tile));

  for (SizeType j = 0; j < tile.size().cols(); ++j)
    for (SizeType i = 0; i < tile.size().rows(); ++i) {
      EXPECT_EQ(memory_view(elIndex(i, j, ld)), tile.ptr(TileElementIndex(i, j)));
    }
}

TYPED_TEST(TileTest, MoveConstructorConst) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  TileElementSize size(m, n);
  Tile<const Type, Device::CPU> const_tile0(size, memory_view, ld);

  Tile<const Type, Device::CPU> const_tile(std::move(const_tile0));
  EXPECT_EQ(TileSizes({0, 0}, 1), getSizes(const_tile0));
  EXPECT_EQ(TileSizes(size, ld), getSizes(const_tile));

  for (SizeType j = 0; j < const_tile.size().cols(); ++j)
    for (SizeType i = 0; i < const_tile.size().rows(); ++i) {
      EXPECT_EQ(memory_view(elIndex(i, j, ld)), const_tile.ptr(TileElementIndex(i, j)));
    }
}

TYPED_TEST(TileTest, MoveConstructorMix) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  TileElementSize size(m, n);
  Tile<Type, Device::CPU> tile0(size, memory_view, ld);

  Tile<const Type, Device::CPU> const_tile(std::move(tile0));
  EXPECT_EQ(TileSizes({0, 0}, 1), getSizes(tile0));
  EXPECT_EQ(TileSizes(size, ld), getSizes(const_tile));

  for (SizeType j = 0; j < const_tile.size().cols(); ++j)
    for (SizeType i = 0; i < const_tile.size().rows(); ++i) {
      EXPECT_EQ(memory_view(elIndex(i, j, ld)), const_tile.ptr(TileElementIndex(i, j)));
    }
}

TYPED_TEST(TileTest, MoveAssignement) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  TileElementSize size(m, n);
  Tile<Type, Device::CPU> tile0(size, memory_view, ld);
  Tile<Type, Device::CPU> tile({1, 1}, memory::MemoryView<Type, Device::CPU>(1), 1);

  tile = std::move(tile0);
  EXPECT_EQ(TileSizes({0, 0}, 1), getSizes(tile0));
  EXPECT_EQ(TileSizes(size, ld), getSizes(tile));

  for (SizeType j = 0; j < tile.size().cols(); ++j)
    for (SizeType i = 0; i < tile.size().rows(); ++i) {
      EXPECT_EQ(memory_view(elIndex(i, j, ld)), tile.ptr(TileElementIndex(i, j)));
    }
}

TYPED_TEST(TileTest, MoveAssignementConst) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  TileElementSize size(m, n);
  Tile<const Type, Device::CPU> const_tile0(size, memory_view, ld);
  Tile<const Type, Device::CPU> const_tile({1, 1}, memory::MemoryView<Type, Device::CPU>(1), 1);

  const_tile = std::move(const_tile0);
  EXPECT_EQ(TileSizes({0, 0}, 1), getSizes(const_tile0));
  EXPECT_EQ(TileSizes(size, ld), getSizes(const_tile));

  for (SizeType j = 0; j < const_tile.size().cols(); ++j)
    for (SizeType i = 0; i < const_tile.size().rows(); ++i) {
      EXPECT_EQ(memory_view(elIndex(i, j, ld)), const_tile.ptr(TileElementIndex(i, j)));
    }
}

TYPED_TEST(TileTest, MoveAssignementMix) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  TileElementSize size(m, n);
  Tile<Type, Device::CPU> tile0(size, memory_view, ld);
  Tile<const Type, Device::CPU> const_tile({1, 1}, memory::MemoryView<Type, Device::CPU>(1), 1);

  const_tile = std::move(tile0);
  EXPECT_EQ(TileSizes({0, 0}, 1), getSizes(tile0));
  EXPECT_EQ(TileSizes(size, ld), getSizes(const_tile));

  for (SizeType j = 0; j < const_tile.size().cols(); ++j)
    for (SizeType i = 0; i < const_tile.size().rows(); ++i) {
      EXPECT_EQ(memory_view(elIndex(i, j, ld)), const_tile.ptr(TileElementIndex(i, j)));
    }
}

TYPED_TEST(TileTest, ReferenceMix) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  TileElementSize size(m, n);
  Tile<Type, Device::CPU> tile0(size, memory_view, ld);
  Tile<const Type, Device::CPU>& const_tile = tile0;

  EXPECT_EQ(TileSizes(size, ld), getSizes(tile0));
  EXPECT_EQ(TileSizes(size, ld), getSizes(const_tile));

  for (SizeType j = 0; j < const_tile.size().cols(); ++j)
    for (SizeType i = 0; i < const_tile.size().rows(); ++i) {
      EXPECT_EQ(memory_view(elIndex(i, j, ld)), const_tile.ptr(TileElementIndex(i, j)));
    }
}

TYPED_TEST(TileTest, PointerMix) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  TileElementSize size(m, n);
  Tile<Type, Device::CPU> tile0(size, memory_view, ld);
  Tile<const Type, Device::CPU>* const_tile = &tile0;

  EXPECT_EQ(TileSizes(size, ld), getSizes(tile0));
  EXPECT_EQ(TileSizes(size, ld), getSizes(*const_tile));

  for (SizeType j = 0; j < const_tile->size().cols(); ++j)
    for (SizeType i = 0; i < const_tile->size().rows(); ++i) {
      EXPECT_EQ(memory_view(elIndex(i, j, ld)), const_tile->ptr(TileElementIndex(i, j)));
    }
}

TYPED_TEST(TileTest, PromiseToFuture) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  TileElementSize size(m, n);
  Tile<Type, Device::CPU> tile(size, memory_view, ld);

  hpx::promise<Tile<Type, Device::CPU>> tile_promise;
  hpx::future<Tile<Type, Device::CPU>> tile_future = tile_promise.get_future();
  tile.setPromise(std::move(tile_promise));
  EXPECT_EQ(false, tile_future.is_ready());

  {
    Tile<Type, Device::CPU> tile1 = std::move(tile);
    EXPECT_EQ(false, tile_future.is_ready());
    EXPECT_EQ(TileSizes({0, 0}, 1), getSizes(tile));
  }

  ASSERT_EQ(true, tile_future.is_ready());
  Tile<Type, Device::CPU> tile2 = std::move(tile_future.get());
  EXPECT_EQ(TileSizes(size, ld), getSizes(tile2));

  for (SizeType j = 0; j < tile2.size().cols(); ++j)
    for (SizeType i = 0; i < tile2.size().rows(); ++i) {
      EXPECT_EQ(memory_view(elIndex(i, j, ld)), tile2.ptr(TileElementIndex(i, j)));
    }
}

TYPED_TEST(TileTest, PromiseToFutureConst) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  TileElementSize size(m, n);
  Tile<Type, Device::CPU> tile(size, memory_view, ld);

  hpx::promise<Tile<Type, Device::CPU>> tile_promise;
  hpx::future<Tile<Type, Device::CPU>> tile_future = tile_promise.get_future();
  tile.setPromise(std::move(tile_promise));
  EXPECT_EQ(false, tile_future.is_ready());

  {
    Tile<const Type, Device::CPU> const_tile = std::move(tile);
    EXPECT_EQ(false, tile_future.is_ready());
    EXPECT_EQ(TileSizes({0, 0}, 1), getSizes(const_tile));
  }

  ASSERT_EQ(true, tile_future.is_ready());
  Tile<Type, Device::CPU> tile2 = std::move(tile_future.get());
  EXPECT_EQ(TileSizes(size, ld), getSizes(tile2));

  for (SizeType j = 0; j < tile2.size().cols(); ++j)
    for (SizeType i = 0; i < tile2.size().rows(); ++i) {
      EXPECT_EQ(memory_view(elIndex(i, j, ld)), tile2.ptr(TileElementIndex(TileElementIndex(i, j))));
    }
}
