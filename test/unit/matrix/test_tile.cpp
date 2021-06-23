//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix/tile.h"

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
using namespace dlaf::test;
using namespace testing;

const std::vector<SizeType> sizes({0, 1, 13, 32});
constexpr SizeType m = 37;
constexpr SizeType n = 87;
constexpr SizeType ld = 133;

SizeType elIndex(TileElementIndex index, SizeType ld) {
  return index.row() + ld * index.col();
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
        memory::MemoryView<Type, Device::CPU> memory_view(ld * n);
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
        memory::MemoryView<Type, Device::CPU> memory_view(ld * n);
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
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

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
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

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
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

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
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

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
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

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
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

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
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

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
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

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
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

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
  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

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
  SizeType m = 37;
  SizeType n = 87;
  SizeType ld = 133;

  memory::MemoryView<TypeParam, Device::CPU> memory_view(ld * n);
  auto mem_view = memory_view;

  TileElementSize size(m, n);
  Tile<TypeParam, Device::CPU> tile(size, std::move(mem_view), ld);

  auto tile_data = dlaf::common::make_data(tile);

  EXPECT_EQ(tile.ptr({0, 0}), data_pointer(tile_data));
  EXPECT_EQ(tile.size().cols(), data_nblocks(tile_data));
  EXPECT_EQ(tile.size().rows(), data_blocksize(tile_data));
  EXPECT_EQ(tile.ld(), data_stride(tile_data));
}

template <class T, Device D>
auto createTileAndPtrChecker(TileElementSize size, SizeType ld) {
  memory::MemoryView<T, D> memory_view(ld * size.cols());
  auto memory_view2 = memory_view;
  Tile<T, D> tile(size, std::move(memory_view), ld);
  // construct a second tile referencing the same memory for testing pointers
  Tile<T, D> tile2(size, std::move(memory_view2), ld);
  auto tile_ptr = [tile2 = std::move(tile2)](const TileElementIndex& index) { return tile2.ptr(index); };
  CHECK_TILE_PTR(tile_ptr, tile);
  return std::make_tuple(std::move(tile), std::move(tile_ptr));
}

template <class T, Device D>
auto createTileChain() {
  using T0 = std::remove_const_t<T>;
  // set up tile chain
  hpx::lcos::local::promise<Tile<T0, D>> tile_p;
  hpx::future<Tile<T0, D>> tmp_tile_f = tile_p.get_future();
  hpx::lcos::local::promise<Tile<T0, D>> next_tile_p;
  hpx::future<Tile<T0, D>> next_tile_f = next_tile_p.get_future();

  hpx::future<Tile<T, D>> tile_f =
      tmp_tile_f.then(hpx::launch::sync,
                      hpx::util::unwrapping([p = std::move(next_tile_p)](auto tile) mutable {
                        tile.setPromise(std::move(p));
                        return Tile<T, D>(std::move(tile));
                      }));

  return std::make_tuple(std::move(tile_p), std::move(tile_f), std::move(next_tile_f));
}

template <class F, class T>
void checkSubtile(F&& ptr, T&& tile, TileElementIndex origin, TileElementSize size) {
  auto subtile_ptr = [&ptr, origin](const TileElementIndex& index) {
    return ptr(index + (origin - TileElementIndex(0, 0)));
  };
  EXPECT_EQ(size, tile.size());
  CHECK_TILE_PTR(subtile_ptr, tile);
}

template <class F, class T>
void checkFullTile(F&& ptr, T&& tile, TileElementSize size) {
  EXPECT_EQ(size, tile.size());
  CHECK_TILE_PTR(ptr, tile);
}

template <class T, Device D>
void testSubtileConst(std::string name, TileElementSize size, SizeType ld,
                      const std::vector<SubTileSpec>& subs, std::size_t last_dep) {
  SCOPED_TRACE(name);
  ASSERT_LE(last_dep, subs.size());

  auto tmp = createTileAndPtrChecker<T, D>(size, ld);
  auto tile = std::move(std::get<0>(tmp));
  auto tile_ptr = std::move(std::get<1>(tmp));

  hpx::lcos::local::promise<Tile<T, D>> tile_p;
  hpx::shared_future<Tile<const T, D>> tile_sf;
  hpx::future<Tile<T, D>> next_tile_f;
  std::tie(tile_p, tile_sf, next_tile_f) = createTileChain<const T, D>();
  ASSERT_TRUE(tile_sf.valid() && !tile_sf.is_ready());
  ASSERT_TRUE(next_tile_f.valid() && !next_tile_f.is_ready());

  // create subtiles
  auto subtiles = splitTile(tile_sf, subs);
  EXPECT_EQ(subs.size(), subtiles.size());

  // append the full tile to the ed of the subtile vector.
  subtiles.emplace_back(std::move(tile_sf));
  ASSERT_FALSE(tile_sf.valid());

  for (const auto& subtile : subtiles) {
    EXPECT_TRUE(subtile.valid());
    EXPECT_FALSE(subtile.is_ready());
  }
  EXPECT_FALSE(next_tile_f.is_ready());

  // Make subtiles ready and check them
  tile_p.set_value(std::move(tile));
  for (const auto& subtile : subtiles) {
    EXPECT_TRUE(subtile.is_ready());
  }
  EXPECT_FALSE(next_tile_f.is_ready());

  // return the origin of the subtile or (0, 0) for the original tile.
  auto get_origin = [&subs](std::size_t i) {
    if (i < subs.size())
      return subs[i].origin;
    return TileElementIndex(0, 0);
  };
  // return the size of the subtile or the size of the original tile.
  auto get_size = [&subs, size](std::size_t i) {
    if (i < subs.size())
      return subs[i].size;
    return size;
  };

  // Check pointer of subtiles (all apart from last_dep) and clear them.
  // As one subtile (last_dep) is still alive next_tile is still locked.
  for (std::size_t i = 0; i < subtiles.size(); ++i) {
    if (i != last_dep) {
      checkSubtile(tile_ptr, subtiles[i].get(), get_origin(i), get_size(i));
      subtiles[i] = {};
    }
  }
  EXPECT_TRUE(subtiles[last_dep].valid());
  EXPECT_TRUE(subtiles[last_dep].is_ready());
  EXPECT_FALSE(next_tile_f.is_ready());

  // Check pointer of last_dep subtile and clear it.
  // next_tile_f should be ready.
  {
    std::size_t i = last_dep;
    checkSubtile(tile_ptr, subtiles[i].get(), get_origin(i), get_size(i));
    subtiles[i] = {};
  }
  EXPECT_TRUE(next_tile_f.is_ready());
  checkFullTile(tile_ptr, next_tile_f.get(), size);
}

TYPED_TEST(TileTest, SubtileConst) {
  using Type = TypeParam;

  testSubtileConst<Type, Device::CPU>("Test Empty", {5, 7}, 8, {}, 0);
  testSubtileConst<Type, Device::CPU>("Test 1", {5, 7}, 8, {{{3, 4}, {2, 3}}}, 1);
  testSubtileConst<Type, Device::CPU>("Test 2", {5, 7}, 8, {{{4, 3}, {0, 0}}, {{4, 6}, {1, 1}}}, 2);
  testSubtileConst<Type, Device::CPU>("Test 3", {5, 7}, 8,
                                      {{{5, 7}, {0, 0}},
                                       {{2, 2}, {2, 2}},
                                       {{3, 0}, {2, 7}},
                                       {{0, 0}, {5, 7}}},
                                      2);
  testSubtileConst<Type, Device::CPU>("Test 4", {5, 7}, 8,
                                      {{{5, 7}, {0, 0}}, {{5, 4}, {0, 3}}, {{2, 7}, {3, 0}}}, 1);
}

template <class T, Device D>
void testSubtile(std::string name, TileElementSize size, SizeType ld,
                 const std::vector<SubTileSpec>& subs, std::size_t last_dep) {
  SCOPED_TRACE(name);
  if (subs.size() > 0) {
    ASSERT_LT(last_dep, subs.size());
  }

  auto tmp = createTileAndPtrChecker<T, D>(size, ld);
  auto tile = std::move(std::get<0>(tmp));
  auto tile_ptr = std::move(std::get<1>(tmp));

  hpx::lcos::local::promise<Tile<T, D>> tile_p;
  hpx::future<Tile<T, D>> tile_f;
  hpx::future<Tile<T, D>> next_tile_f;
  std::tie(tile_p, tile_f, next_tile_f) = createTileChain<T, D>();
  ASSERT_TRUE(tile_f.valid() && !tile_f.is_ready());
  ASSERT_TRUE(next_tile_f.valid() && !next_tile_f.is_ready());

  // create subtiles
  auto subtiles = splitTileDisjoint(tile_f, subs);
  ASSERT_TRUE(tile_f.valid());
  EXPECT_FALSE(tile_f.is_ready());
  EXPECT_EQ(subs.size(), subtiles.size());

  for (const auto& subtile : subtiles) {
    EXPECT_TRUE(subtile.valid());
    EXPECT_FALSE(subtile.is_ready());
  }
  EXPECT_FALSE(tile_f.is_ready());
  EXPECT_FALSE(next_tile_f.is_ready());

  // Make subtiles ready and check them
  tile_p.set_value(std::move(tile));

  if (subtiles.size() > 0) {
    for (const auto& subtile : subtiles) {
      EXPECT_TRUE(subtile.is_ready());
    }
    EXPECT_FALSE(tile_f.is_ready());
    EXPECT_FALSE(next_tile_f.is_ready());

    // return the origin of the subtile.
    auto get_origin = [&subs](std::size_t i) { return subs[i].origin; };
    // return the size of the subtile.
    auto get_size = [&subs](std::size_t i) { return subs[i].size; };

    // Check pointer of subtiles (all apart from last_dep) and clear them.
    // As one subtile (last_dep) is still alive next_tile is still locked.
    for (std::size_t i = 0; i < subtiles.size(); ++i) {
      if (i != last_dep) {
        checkSubtile(tile_ptr, subtiles[i].get(), get_origin(i), get_size(i));
      }
    }
    EXPECT_TRUE(subtiles[last_dep].valid());
    EXPECT_TRUE(subtiles[last_dep].is_ready());
    EXPECT_FALSE(tile_f.is_ready());
    EXPECT_FALSE(next_tile_f.is_ready());

    // Check pointer of last_dep subtile and clear it.
    // tile_f should be ready.
    {
      std::size_t i = last_dep;
      checkSubtile(tile_ptr, subtiles[i].get(), get_origin(i), get_size(i));
    }
  }
  EXPECT_TRUE(tile_f.is_ready());
  EXPECT_FALSE(next_tile_f.is_ready());
  // check tile pointer and unlock next_tile.
  // next_tile_f should be ready.
  checkFullTile(tile_ptr, tile_f.get(), size);

  EXPECT_TRUE(next_tile_f.is_ready());
  checkFullTile(tile_ptr, next_tile_f.get(), size);
}

TYPED_TEST(TileTest, Subtile) {
  using Type = TypeParam;

  testSubtile<Type, Device::CPU>("Test Empty", {5, 7}, 8, {}, 0);
  testSubtile<Type, Device::CPU>("Test 1", {5, 7}, 8, {{{3, 4}, {2, 3}}}, 0);
  testSubtile<Type, Device::CPU>("Test 2", {5, 7}, 8, {{{4, 3}, {0, 0}}, {{4, 6}, {1, 1}}}, 1);
  testSubtile<Type, Device::CPU>("Test 3", {5, 7}, 8,
                                 {{{5, 7}, {0, 0}},
                                  {{1, 2}, {2, 2}},
                                  {{3, 0}, {2, 7}},
                                  {{0, 0}, {1, 4}},
                                  {{0, 4}, {3, 3}}},
                                 2);
  testSubtile<Type, Device::CPU>("Test 4", {5, 7}, 8,
                                 {{{5, 7}, {0, 0}}, {{5, 4}, {0, 3}}, {{2, 7}, {3, 0}}}, 1);
}
