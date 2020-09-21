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

#include <gtest/gtest.h>
#include <hpx/local/future.hpp>

#include "dlaf/matrix/index.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf_test;
using namespace testing;

const std::vector<SizeType> sizes({0, 1, 13, 32});
constexpr SizeType m = 37;
constexpr SizeType n = 87;
constexpr SizeType ld = 133;

std::size_t elIndex(TileElementIndex index, SizeType ld) {
  using util::size_t::sum;
  using util::size_t::mul;
  return sum(index.row(), mul(ld, index.col()));
}

using TileSizes = std::tuple<TileElementSize, SizeType>;

template <class T, Device device>
TileSizes getSizes(const Tile<T, device>& tile) {
  return TileSizes(tile.size(), tile.ld());
}

template <typename Type>
class TileTest : public ::testing::Test {};

TYPED_TEST_SUITE(TileTest, MatrixElementTypes);

TYPED_TEST(TileTest, Constructor) {
  using Type = TypeParam;

  auto el = [](const TileElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(10. * i + 0.01 * j, j / 2. - 0.01 * i);
  };
  auto el2 = [](const TileElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(-10. * i + 0.02 * j, j + 0.01 * i);
  };

  for (const auto m : sizes) {
    for (const auto n : sizes) {
      SizeType min_ld = std::max<SizeType>(1, m);
      for (const SizeType ld : {min_ld, min_ld + 64}) {
        memory::MemoryView<Type, Device::CPU> memory_view(util::size_t::mul(ld, n));
        TileElementSize size(m, n);
        for (SizeType j = 0; j < size.cols(); ++j) {
          for (SizeType i = 0; i < size.rows(); ++i) {
            TileElementIndex index(i, j);
            *memory_view(elIndex(index, ld)) = el(index);
          }
        }

        auto mem_view = memory_view;  // Copy the memory view to check the elements later.
        Tile<Type, Device::CPU> tile(size, std::move(mem_view), ld);
        EXPECT_EQ(TileSizes(size, ld), getSizes(tile));

        CHECK_TILE_EQ(el, tile);

        auto ptr = [&memory_view, ld](const TileElementIndex& index) {
          return memory_view(elIndex(index, ld));
        };
        CHECK_TILE_PTR(ptr, tile);

        set(tile, el2);

        for (SizeType j = 0; j < size.cols(); ++j) {
          for (SizeType i = 0; i < size.rows(); ++i) {
            TileElementIndex index(i, j);
            EXPECT_EQ(el2(index), *memory_view(elIndex(index, ld)));
          }
        }
      }
    }
  }
}

TYPED_TEST(TileTest, ConstructorConst) {
  using Type = TypeParam;

  auto el = [](const TileElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(10. * i + 0.01 * j, j / 2. - 0.01 * i);
  };

  for (const auto m : sizes) {
    for (const auto n : sizes) {
      SizeType min_ld = std::max<SizeType>(1, m);
      for (const SizeType ld : {min_ld, min_ld + 64}) {
        memory::MemoryView<Type, Device::CPU> memory_view(util::size_t::mul(ld, n));
        TileElementSize size(m, n);
        for (SizeType j = 0; j < size.cols(); ++j) {
          for (SizeType i = 0; i < size.rows(); ++i) {
            TileElementIndex index(i, j);
            *memory_view(elIndex(index, ld)) = el(index);
          }
        }

        auto mem_view = memory_view;  // Copy the memory view to check the elements later.
        Tile<const Type, Device::CPU> tile(size, std::move(mem_view), ld);
        EXPECT_EQ(TileSizes(size, ld), getSizes(tile));

        CHECK_TILE_EQ(el, tile);

        auto ptr = [&memory_view, ld](const TileElementIndex& index) {
          return memory_view(elIndex(index, ld));
        };
        CHECK_TILE_PTR(ptr, tile);
      }
    }
  }
}

TYPED_TEST(TileTest, MoveConstructor) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(util::size_t::mul(ld, n));

  TileElementSize size(m, n);
  auto mem_view = memory_view;  // Copy the memory view to check the elements later.
  Tile<Type, Device::CPU> tile0(size, std::move(mem_view), ld);

  Tile<Type, Device::CPU> tile(std::move(tile0));
  EXPECT_EQ(TileSizes({0, 0}, 1), getSizes(tile0));
  EXPECT_EQ(TileSizes(size, ld), getSizes(tile));

  auto ptr = [&memory_view](const TileElementIndex& index) { return memory_view(elIndex(index, ld)); };
  CHECK_TILE_PTR(ptr, tile);
}

TYPED_TEST(TileTest, MoveConstructorConst) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(util::size_t::mul(ld, n));

  TileElementSize size(m, n);
  auto mem_view = memory_view;  // Copy the memory view to check the elements later.
  Tile<const Type, Device::CPU> const_tile0(size, std::move(mem_view), ld);

  Tile<const Type, Device::CPU> const_tile(std::move(const_tile0));
  EXPECT_EQ(TileSizes({0, 0}, 1), getSizes(const_tile0));
  EXPECT_EQ(TileSizes(size, ld), getSizes(const_tile));

  auto ptr = [&memory_view](const TileElementIndex& index) { return memory_view(elIndex(index, ld)); };
  CHECK_TILE_PTR(ptr, const_tile);
}

TYPED_TEST(TileTest, MoveConstructorMix) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(util::size_t::mul(ld, n));

  TileElementSize size(m, n);
  auto mem_view = memory_view;  // Copy the memory view to check the elements later.
  Tile<Type, Device::CPU> tile0(size, std::move(mem_view), ld);

  Tile<const Type, Device::CPU> const_tile(std::move(tile0));
  EXPECT_EQ(TileSizes({0, 0}, 1), getSizes(tile0));
  EXPECT_EQ(TileSizes(size, ld), getSizes(const_tile));

  auto ptr = [&memory_view](const TileElementIndex& index) { return memory_view(elIndex(index, ld)); };
  CHECK_TILE_PTR(ptr, const_tile);
}

TYPED_TEST(TileTest, MoveAssignement) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(util::size_t::mul(ld, n));

  TileElementSize size(m, n);
  auto mem_view = memory_view;  // Copy the memory view to check the elements later.
  Tile<Type, Device::CPU> tile0(size, std::move(mem_view), ld);
  Tile<Type, Device::CPU> tile({1, 1}, memory::MemoryView<Type, Device::CPU>(1), 1);

  tile = std::move(tile0);
  EXPECT_EQ(TileSizes({0, 0}, 1), getSizes(tile0));
  EXPECT_EQ(TileSizes(size, ld), getSizes(tile));

  auto ptr = [&memory_view](const TileElementIndex& index) { return memory_view(elIndex(index, ld)); };
  CHECK_TILE_PTR(ptr, tile);
}

TYPED_TEST(TileTest, MoveAssignementConst) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(util::size_t::mul(ld, n));

  TileElementSize size(m, n);
  auto mem_view = memory_view;  // Copy the memory view to check the elements later.
  Tile<const Type, Device::CPU> const_tile0(size, std::move(mem_view), ld);
  Tile<const Type, Device::CPU> const_tile({1, 1}, memory::MemoryView<Type, Device::CPU>(1), 1);

  const_tile = std::move(const_tile0);
  EXPECT_EQ(TileSizes({0, 0}, 1), getSizes(const_tile0));
  EXPECT_EQ(TileSizes(size, ld), getSizes(const_tile));

  auto ptr = [&memory_view](const TileElementIndex& index) { return memory_view(elIndex(index, ld)); };
  CHECK_TILE_PTR(ptr, const_tile);
}

TYPED_TEST(TileTest, MoveAssignementMix) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(util::size_t::mul(ld, n));

  TileElementSize size(m, n);
  auto mem_view = memory_view;  // Copy the memory view to check the elements later.
  Tile<Type, Device::CPU> tile0(size, std::move(mem_view), ld);
  Tile<const Type, Device::CPU> const_tile({1, 1}, memory::MemoryView<Type, Device::CPU>(1), 1);

  const_tile = std::move(tile0);
  EXPECT_EQ(TileSizes({0, 0}, 1), getSizes(tile0));
  EXPECT_EQ(TileSizes(size, ld), getSizes(const_tile));

  auto ptr = [&memory_view](const TileElementIndex& index) { return memory_view(elIndex(index, ld)); };
  CHECK_TILE_PTR(ptr, const_tile);
}

TYPED_TEST(TileTest, ReferenceMix) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(util::size_t::mul(ld, n));

  TileElementSize size(m, n);
  auto mem_view = memory_view;  // Copy the memory view to check the elements later.
  Tile<Type, Device::CPU> tile0(size, std::move(mem_view), ld);
  Tile<const Type, Device::CPU>& const_tile = tile0;

  EXPECT_EQ(TileSizes(size, ld), getSizes(tile0));
  EXPECT_EQ(TileSizes(size, ld), getSizes(const_tile));

  auto ptr = [&memory_view](const TileElementIndex& index) { return memory_view(elIndex(index, ld)); };
  CHECK_TILE_PTR(ptr, const_tile);
}

TYPED_TEST(TileTest, PointerMix) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(util::size_t::mul(ld, n));

  TileElementSize size(m, n);
  auto mem_view = memory_view;  // Copy the memory view to check the elements later.
  Tile<Type, Device::CPU> tile0(size, std::move(mem_view), ld);
  Tile<const Type, Device::CPU>* const_tile = &tile0;

  EXPECT_EQ(TileSizes(size, ld), getSizes(tile0));
  EXPECT_EQ(TileSizes(size, ld), getSizes(*const_tile));

  auto ptr = [&memory_view](const TileElementIndex& index) { return memory_view(elIndex(index, ld)); };
  CHECK_TILE_PTR(ptr, *const_tile);
}

TYPED_TEST(TileTest, PromiseToFuture) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(util::size_t::mul(ld, n));

  TileElementSize size(m, n);
  auto mem_view = memory_view;  // Copy the memory view to check the elements later.
  Tile<Type, Device::CPU> tile(size, std::move(mem_view), ld);

  hpx::lcos::local::promise<Tile<Type, Device::CPU>> tile_promise;
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

  auto ptr = [&memory_view](const TileElementIndex& index) { return memory_view(elIndex(index, ld)); };
  CHECK_TILE_PTR(ptr, tile2);
}

TYPED_TEST(TileTest, PromiseToFutureConst) {
  using Type = TypeParam;
  memory::MemoryView<Type, Device::CPU> memory_view(util::size_t::mul(ld, n));

  TileElementSize size(m, n);
  auto mem_view = memory_view;  // Copy the memory view to check the elements later.
  Tile<Type, Device::CPU> tile(size, std::move(mem_view), ld);

  hpx::lcos::local::promise<Tile<Type, Device::CPU>> tile_promise;
  hpx::future<Tile<Type, Device::CPU>> tile_future = tile_promise.get_future();
  tile.setPromise(std::move(tile_promise));
  EXPECT_EQ(false, tile_future.is_ready());

  {
    Tile<const Type, Device::CPU> const_tile = std::move(tile);
    EXPECT_EQ(false, tile_future.is_ready());
    EXPECT_EQ(TileSizes({0, 0}, 1), getSizes(tile));
  }

  ASSERT_EQ(true, tile_future.is_ready());
  Tile<Type, Device::CPU> tile2 = std::move(tile_future.get());
  EXPECT_EQ(TileSizes(size, ld), getSizes(tile2));

  auto ptr = [&memory_view](const TileElementIndex& index) { return memory_view(elIndex(index, ld)); };
  CHECK_TILE_PTR(ptr, tile2);
}

// Buffer
TYPED_TEST(TileTest, CreateBuffer) {
  using namespace dlaf;

  SizeType m = 37;
  SizeType n = 87;
  SizeType ld = 133;

  memory::MemoryView<TypeParam, Device::CPU> memory_view(util::size_t::mul(ld, n));
  auto mem_view = memory_view;

  TileElementSize size(m, n);
  Tile<TypeParam, Device::CPU> tile(size, std::move(mem_view), ld);

  auto tile_data = dlaf::common::make_data(tile);

  EXPECT_EQ(tile.ptr({0, 0}), data_pointer(tile_data));
  EXPECT_EQ(tile.size().cols(), data_nblocks(tile_data));
  EXPECT_EQ(tile.size().rows(), data_blocksize(tile_data));
  EXPECT_EQ(tile.ld(), data_stride(tile_data));
}
