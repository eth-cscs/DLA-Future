//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix/tile.h"

#include <stdexcept>

#include <gtest/gtest.h>
#include <pika/future.hpp>
#include <pika/unwrap.hpp>

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
  using TileType = Tile<Type, Device::CPU>;
  using TileDataType = typename TileType::TileDataType;

  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  TileElementSize size(m, n);
  auto mem_view = memory_view;  // Copy the memory view to check the elements later.
  TileType tile(size, std::move(mem_view), ld);

  pika::lcos::local::promise<TileDataType> tile_promise;
  auto tile_future = tile_promise.get_future();
  tile.setPromise(std::move(tile_promise));
  EXPECT_EQ(false, tile_future.is_ready());

  {
    TileType tile1 = std::move(tile);
    EXPECT_EQ(false, tile_future.is_ready());
    EXPECT_EQ(TileSizes({0, 0}, 1), getSizes(tile));
  }

  ASSERT_EQ(true, tile_future.is_ready());
  TileType tile2{tile_future.get()};
  EXPECT_EQ(TileSizes(size, ld), getSizes(tile2));

  auto ptr = [&memory_view](const TileElementIndex& index) { return memory_view(elIndex(index, ld)); };
  CHECK_TILE_PTR(ptr, tile2);
}

TYPED_TEST(TileTest, PromiseToFutureConst) {
  using Type = TypeParam;
  using TileType = Tile<Type, Device::CPU>;
  using ConstTileType = Tile<const Type, Device::CPU>;
  using TileDataType = typename TileType::TileDataType;

  memory::MemoryView<Type, Device::CPU> memory_view(ld * n);

  TileElementSize size(m, n);
  auto mem_view = memory_view;  // Copy the memory view to check the elements later.
  TileType tile(size, std::move(mem_view), ld);

  pika::lcos::local::promise<TileDataType> tile_promise;
  auto tile_future = tile_promise.get_future();
  tile.setPromise(std::move(tile_promise));
  EXPECT_EQ(false, tile_future.is_ready());

  {
    ConstTileType const_tile = std::move(tile);
    EXPECT_EQ(false, tile_future.is_ready());
    EXPECT_EQ(TileSizes({0, 0}, 1), getSizes(tile));
  }

  ASSERT_EQ(true, tile_future.is_ready());
  TileType tile2{tile_future.get()};
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
  typename Tile<T, D>::TileDataType tile(size, std::move(memory_view), ld);
  // construct a second tile referencing the same memory for testing pointers
  Tile<T, D> tile2(size, std::move(memory_view2), ld);
  auto tile_ptr = [tile2 = std::move(tile2)](const TileElementIndex& index) { return tile2.ptr(index); };
  return std::make_tuple(std::move(tile), std::move(tile_ptr));
}

template <class T, Device D>
auto createTileChain() {
  using TileType = Tile<T, Device::CPU>;
  using NonConstTileType = typename TileType::TileType;
  using TileDataType = typename TileType::TileDataType;

  // set up tile chain
  pika::lcos::local::promise<TileDataType> tile_p;
  auto tmp_tile_f = tile_p.get_future();
  pika::lcos::local::promise<TileDataType> next_tile_p;
  auto next_tile_f = next_tile_p.get_future();

  pika::future<TileType> tile_f =
      tmp_tile_f.then(pika::launch::sync,
                      pika::unwrapping([p = std::move(next_tile_p)](auto tile) mutable {
                        return TileType(
                            std::move(NonConstTileType(std::move(tile)).setPromise(std::move(p))));
                      }));

  return std::make_tuple(std::move(tile_p), std::move(tile_f), std::move(next_tile_f));
}

template <class F, class T>
void checkSubtile(F&& ptr, T&& tile, SubTileSpec spec) {
  auto subtile_ptr = [&ptr, origin = spec.origin](const TileElementIndex& index) {
    return ptr(index + common::sizeFromOrigin(origin));
  };
  EXPECT_EQ(spec.size, tile.size());
  CHECK_TILE_PTR(subtile_ptr, tile);
}

template <class F, class T>
void checkFullTile(F&& ptr, T&& tile, TileElementSize size) {
  EXPECT_EQ(size, tile.size());
  CHECK_TILE_PTR(ptr, tile);
}

// TileFutureOrConstTileSharedFuture should be
// either pika::future<Tile<T, D>> or pika::shared_future<Tile<const T, D>>
template <class TileFutureOrConstTileSharedFuture>
void checkValidNonReady(const std::vector<TileFutureOrConstTileSharedFuture>& subtiles) {
  for (const auto& subtile : subtiles) {
    EXPECT_TRUE(subtile.valid());
    EXPECT_FALSE(subtile.is_ready());
  }
}

// TileFutureOrConstTileSharedFuture should be
// either pika::future<Tile<T, D>> or pika::shared_future<Tile<const T, D>>
// TileFuture should be pika::future<Tile<T, D>>
template <class F, class TileFutureOrConstTileSharedFuture, class TileFuture>
void checkReadyAndDependencyChain(F&& tile_ptr, std::vector<TileFutureOrConstTileSharedFuture>& subtiles,
                                  const std::vector<SubTileSpec>& specs, std::size_t last_dep,
                                  TileFuture& next_tile_f) {
  ASSERT_EQ(subtiles.size(), specs.size());
  ASSERT_GT(subtiles.size(), last_dep);

  for (const auto& subtile : subtiles) {
    EXPECT_TRUE(subtile.is_ready());
  }
  EXPECT_FALSE(next_tile_f.is_ready());

  // Check pointer of subtiles (all apart from last_dep) and clear them.
  // As one subtile (last_dep) is still alive next_tile is still locked.
  for (std::size_t i = 0; i < subtiles.size(); ++i) {
    if (i != last_dep) {
      checkSubtile(tile_ptr, subtiles[i].get(), specs[i]);
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
    checkSubtile(tile_ptr, subtiles[i].get(), specs[i]);
    subtiles[i] = {};
  }
  EXPECT_TRUE(next_tile_f.is_ready());
}

template <class T, Device D>
void testSubtileConst(std::string name, TileElementSize size, SizeType ld, const SubTileSpec& spec,
                      std::size_t last_dep) {
  SCOPED_TRACE(name);
  ASSERT_LE(last_dep, 1);

  auto [tile, tile_ptr] = createTileAndPtrChecker<T, D>(size, ld);

  auto [tile_p, tile_f, next_tile_f] = createTileChain<const T, D>();
  auto tile_sf = tile_f.share();
  ASSERT_TRUE(tile_sf.valid() && !tile_sf.is_ready());
  ASSERT_TRUE(next_tile_f.valid() && !next_tile_f.is_ready());

  // create subtiles
  auto subtile = splitTile(tile_sf, spec);

  // append the full tile to the end of the subtile vector and add its specs to full_specs.
  std::vector<pika::shared_future<Tile<const T, D>>> subtiles = {std::move(subtile), std::move(tile_sf)};
  std::vector<SubTileSpec> full_specs = {spec, {{0, 0}, size}};

  checkValidNonReady(subtiles);
  ASSERT_FALSE(next_tile_f.is_ready());

  // Make subtiles ready
  tile_p.set_value(std::move(tile));

  checkReadyAndDependencyChain(tile_ptr, subtiles, full_specs, last_dep, next_tile_f);

  // Check next tile in the dependency chain
  checkFullTile(tile_ptr, Tile<T, D>{next_tile_f.get()}, size);
}

template <class T, Device D>
void testSubtilesConst(std::string name, TileElementSize size, SizeType ld,
                       std::vector<SubTileSpec> specs, std::size_t last_dep) {
  SCOPED_TRACE(name);
  ASSERT_LE(last_dep, specs.size());

  auto [tile, tile_ptr] = createTileAndPtrChecker<T, D>(size, ld);

  auto [tile_p, tile_f, next_tile_f] = createTileChain<const T, D>();
  auto tile_sf = tile_f.share();
  ASSERT_TRUE(tile_sf.valid() && !tile_sf.is_ready());
  ASSERT_TRUE(next_tile_f.valid() && !next_tile_f.is_ready());

  // create subtiles
  auto subtiles = splitTile(tile_sf, specs);
  ASSERT_EQ(specs.size(), subtiles.size());

  // append the full tile to the end of the subtile vector and add its specs to full_specs.
  subtiles.emplace_back(std::move(tile_sf));
  specs.push_back({{0, 0}, size});

  checkValidNonReady(subtiles);
  ASSERT_FALSE(next_tile_f.is_ready());

  // Make subtiles ready and check them
  tile_p.set_value(std::move(tile));

  checkReadyAndDependencyChain(tile_ptr, subtiles, specs, last_dep, next_tile_f);
  checkFullTile(tile_ptr, Tile<T, D>{next_tile_f.get()}, size);
}

template <class T, Device D>
void testSubOfSubtileConst(std::string name, TileElementSize size, SizeType ld,
                           std::vector<SubTileSpec> specs, const SubTileSpec& subspec,
                           std::size_t last_dep) {
  SCOPED_TRACE(name);
  ASSERT_LE(1, specs.size());  // Need at least a subtile to create a subsubtile
  ASSERT_LE(last_dep, specs.size() + 1);
  // specs.size() -> subsubtile
  // specs.size() + 1 -> full tile

  auto [tile, tile_ptr] = createTileAndPtrChecker<T, D>(size, ld);

  auto [tile_p, tile_f, next_tile_f] = createTileChain<const T, D>();
  auto tile_sf = tile_f.share();
  ASSERT_TRUE(tile_sf.valid() && !tile_sf.is_ready());
  ASSERT_TRUE(next_tile_f.valid() && !next_tile_f.is_ready());

  // create subtiles
  auto subtiles = splitTile(tile_sf, specs);
  ASSERT_EQ(specs.size(), subtiles.size());

  // create sub tile of subtiles[0]
  auto subsubtile = splitTile(subtiles[0], subspec);

  // append the subsubtile and the full tile to the end of the subtile vector and add its specs to full_specs.
  subtiles.emplace_back(std::move(subsubtile));
  subtiles.emplace_back(std::move(tile_sf));
  specs.push_back({specs[0].origin + common::sizeFromOrigin(subspec.origin), subspec.size});
  specs.push_back({{0, 0}, size});

  checkValidNonReady(subtiles);
  ASSERT_FALSE(next_tile_f.is_ready());

  // Make subtiles ready and check them
  tile_p.set_value(std::move(tile));

  checkReadyAndDependencyChain(tile_ptr, subtiles, specs, last_dep, next_tile_f);
  checkFullTile(tile_ptr, Tile<T, D>{next_tile_f.get()}, size);
}

TYPED_TEST(TileTest, SubtileConst) {
  using Type = TypeParam;

  testSubtileConst<Type, Device::CPU>("Test 1", {5, 7}, 8, {{3, 4}, {2, 3}}, 0);
  testSubtileConst<Type, Device::CPU>("Test 2", {5, 7}, 8, {{4, 6}, {1, 1}}, 1);
  testSubtileConst<Type, Device::CPU>("Test 3", {5, 7}, 8, {{0, 0}, {5, 7}}, 0);

  testSubtilesConst<Type, Device::CPU>("Test Vector Empty", {5, 7}, 8, {}, 0);
  testSubtilesConst<Type, Device::CPU>("Test Vector 1", {5, 7}, 8, {{{3, 4}, {2, 3}}}, 1);
  testSubtilesConst<Type, Device::CPU>("Test Vector 2", {5, 7}, 8, {{{4, 3}, {0, 0}}, {{4, 6}, {1, 1}}},
                                       2);
  testSubtilesConst<Type, Device::CPU>("Test Vector 3", {5, 7}, 8,
                                       {{{5, 7}, {0, 0}},
                                        {{2, 2}, {2, 2}},
                                        {{3, 0}, {2, 7}},
                                        {{0, 0}, {5, 7}}},
                                       2);
  testSubtilesConst<Type, Device::CPU>("Test Vector 4", {5, 7}, 8,
                                       {{{5, 7}, {0, 0}}, {{5, 4}, {0, 3}}, {{2, 7}, {3, 0}}}, 1);

  testSubOfSubtileConst<Type, Device::CPU>("Test SubSub 1", {6, 7}, 6,
                                           {{{2, 3}, {3, 3}}, {{4, 6}, {1, 1}}}, {{0, 2}, {2, 1}}, 0);
  testSubOfSubtileConst<Type, Device::CPU>("Test SubSub 2", {6, 7}, 6,
                                           {{{2, 3}, {3, 3}}, {{4, 6}, {1, 1}}}, {{0, 2}, {2, 1}}, 1);
  testSubOfSubtileConst<Type, Device::CPU>("Test SubSub 2", {6, 7}, 6,
                                           {{{2, 3}, {3, 3}}, {{4, 6}, {1, 1}}}, {{0, 2}, {2, 1}}, 2);
  testSubOfSubtileConst<Type, Device::CPU>("Test SubSub 3", {6, 7}, 6,
                                           {{{2, 3}, {3, 3}}, {{4, 6}, {1, 1}}}, {{0, 2}, {2, 1}}, 3);
}

template <class T, Device D>
void testSubtile(std::string name, TileElementSize size, SizeType ld, const SubTileSpec& spec) {
  SCOPED_TRACE(name);

  auto [tile, tile_ptr] = createTileAndPtrChecker<T, D>(size, ld);

  auto [tile_p, tile_f, next_tile_f] = createTileChain<T, D>();
  ASSERT_TRUE(tile_f.valid() && !tile_f.is_ready());
  ASSERT_TRUE(next_tile_f.valid() && !next_tile_f.is_ready());

  // create subtiles
  auto subtile = splitTile(tile_f, spec);
  ASSERT_TRUE(tile_f.valid());
  ASSERT_FALSE(tile_f.is_ready());

  // append the full tile to the end of the subtile vector and add its specs to full_specs.
  std::vector<pika::future<Tile<T, D>>> subtiles;
  subtiles.emplace_back(std::move(subtile));
  std::vector<SubTileSpec> full_specs = {spec};

  checkValidNonReady(subtiles);
  ASSERT_FALSE(tile_f.is_ready());
  ASSERT_FALSE(next_tile_f.is_ready());

  // Make subtiles ready
  tile_p.set_value(std::move(tile));

  checkReadyAndDependencyChain(tile_ptr, subtiles, full_specs, 0, tile_f);

  ASSERT_TRUE(tile_f.is_ready());
  EXPECT_FALSE(next_tile_f.is_ready());
  // check tile pointer and unlock next_tile.
  // next_tile_f should be ready.
  checkFullTile(tile_ptr, tile_f.get(), size);

  ASSERT_TRUE(next_tile_f.is_ready());
  checkFullTile(tile_ptr, Tile<T, D>{next_tile_f.get()}, size);
}

template <class T, Device D>
void testSubtileMove(std::string name, TileElementSize size, SizeType ld, const SubTileSpec& spec) {
  SCOPED_TRACE(name);

  auto [tile, tile_ptr] = createTileAndPtrChecker<T, D>(size, ld);

  auto [tile_p, tile_f, next_tile_f] = createTileChain<T, D>();
  ASSERT_TRUE(tile_f.valid() && !tile_f.is_ready());
  ASSERT_TRUE(next_tile_f.valid() && !next_tile_f.is_ready());

  // create subtiles
  auto subtile = splitTile(std::move(tile_f), spec);

  // append the full tile to the end of the subtile vector and add its specs to full_specs.
  std::vector<pika::future<Tile<T, D>>> subtiles;
  subtiles.emplace_back(std::move(subtile));
  std::vector<SubTileSpec> full_specs = {spec};

  checkValidNonReady(subtiles);
  ASSERT_FALSE(next_tile_f.is_ready());

  // Make subtiles ready
  tile_p.set_value(std::move(tile));

  checkReadyAndDependencyChain(tile_ptr, subtiles, full_specs, 0, next_tile_f);

  ASSERT_TRUE(next_tile_f.is_ready());
  checkFullTile(tile_ptr, Tile<T, D>{next_tile_f.get()}, size);
}

template <class T, Device D>
void testSubtilesDisjoint(std::string name, TileElementSize size, SizeType ld,
                          const std::vector<SubTileSpec>& specs, std::size_t last_dep) {
  SCOPED_TRACE(name);
  if (specs.size() > 0) {
    ASSERT_LT(last_dep, specs.size());
  }

  auto [tile, tile_ptr] = createTileAndPtrChecker<T, D>(size, ld);

  auto [tile_p, tile_f, next_tile_f] = createTileChain<T, D>();
  ASSERT_TRUE(tile_f.valid() && !tile_f.is_ready());
  ASSERT_TRUE(next_tile_f.valid() && !next_tile_f.is_ready());

  // create subtiles
  auto subtiles = splitTileDisjoint(tile_f, specs);
  ASSERT_TRUE(tile_f.valid());
  ASSERT_FALSE(tile_f.is_ready());
  ASSERT_EQ(specs.size(), subtiles.size());

  checkValidNonReady(subtiles);
  ASSERT_FALSE(tile_f.is_ready());
  ASSERT_FALSE(next_tile_f.is_ready());

  // Make subtiles ready and check them
  tile_p.set_value(std::move(tile));

  if (subtiles.size() > 0) {
    checkReadyAndDependencyChain(tile_ptr, subtiles, specs, last_dep, tile_f);
  }
  ASSERT_TRUE(tile_f.is_ready());
  EXPECT_FALSE(next_tile_f.is_ready());
  // check tile pointer and unlock next_tile.
  // next_tile_f should be ready.
  checkFullTile(tile_ptr, tile_f.get(), size);

  ASSERT_TRUE(next_tile_f.is_ready());
  checkFullTile(tile_ptr, Tile<T, D>{next_tile_f.get()}, size);
}

template <class T, Device D>
void testSubOfSubtile(std::string name, TileElementSize size, SizeType ld,
                      std::vector<SubTileSpec> specs, const SubTileSpec& subspec) {
  SCOPED_TRACE(name);
  ASSERT_LE(1, specs.size());  // Need at least a subtile to create a subsubtile
  // last_dep = 0 -> subsubtile

  auto [tile, tile_ptr] = createTileAndPtrChecker<T, D>(size, ld);

  auto [tile_p, tile_f, next_tile_f] = createTileChain<T, D>();
  ASSERT_TRUE(tile_f.valid() && !tile_f.is_ready());
  ASSERT_TRUE(next_tile_f.valid() && !next_tile_f.is_ready());

  // create subtiles
  auto subtiles = splitTileDisjoint(tile_f, specs);
  ASSERT_TRUE(tile_f.valid());
  ASSERT_FALSE(tile_f.is_ready());
  ASSERT_EQ(specs.size(), subtiles.size());

  // extract subtile from which we will create the subsubtile
  auto subtile_f = std::move(subtiles[0]);
  // create subsubtile
  auto subsubtile = splitTile(subtile_f, subspec);
  ASSERT_TRUE(subtile_f.valid());

  // replace the subtile with its subsubtile and update specs
  auto spec0 = specs[0];
  subtiles[0] = std::move(subsubtile);
  specs[0] = {spec0.origin + common::sizeFromOrigin(subspec.origin), subspec.size};

  checkValidNonReady(subtiles);
  ASSERT_FALSE(subtile_f.is_ready());
  ASSERT_FALSE(tile_f.is_ready());
  ASSERT_FALSE(next_tile_f.is_ready());

  // The dependencies are currently in the following way

  // ---> subtiles[0] (the subsubtile) -> subtile_f ---> tile_f -> next_tile_f
  //  |-> subtiles[1] -------------------------------|
  //  |-> subtiles[2] -------------------------------|
  // ...

  // Make subtiles ready and check them
  tile_p.set_value(std::move(tile));

  checkReadyAndDependencyChain(tile_ptr, subtiles, specs, 0, subtile_f);
  ASSERT_TRUE(subtile_f.is_ready());
  EXPECT_FALSE(tile_f.is_ready());
  EXPECT_FALSE(next_tile_f.is_ready());
  // check subtile pointer and unlock tile.
  // tile_f should be ready.
  checkSubtile(tile_ptr, subtile_f.get(), spec0);

  ASSERT_TRUE(tile_f.is_ready());
  EXPECT_FALSE(next_tile_f.is_ready());
  // check tile pointer and unlock next_tile.
  // next_tile_f should be ready.
  checkFullTile(tile_ptr, tile_f.get(), size);

  ASSERT_TRUE(next_tile_f.is_ready());
  checkFullTile(tile_ptr, Tile<T, D>{next_tile_f.get()}, size);
}

TYPED_TEST(TileTest, Subtile) {
  using Type = TypeParam;

  testSubtile<Type, Device::CPU>("Test 1", {5, 7}, 8, {{3, 4}, {2, 3}});
  testSubtile<Type, Device::CPU>("Test 2", {5, 7}, 8, {{4, 6}, {1, 1}});
  testSubtile<Type, Device::CPU>("Test 3", {5, 7}, 8, {{0, 0}, {5, 7}});

  testSubtileMove<Type, Device::CPU>("Test Move 1", {5, 7}, 8, {{3, 4}, {2, 3}});
  testSubtileMove<Type, Device::CPU>("Test Move 2", {5, 7}, 8, {{4, 6}, {1, 1}});
  testSubtileMove<Type, Device::CPU>("Test Move 3", {5, 7}, 8, {{0, 0}, {5, 7}});

  testSubtilesDisjoint<Type, Device::CPU>("Test Vector Empty", {5, 7}, 8, {}, 0);
  testSubtilesDisjoint<Type, Device::CPU>("Test Vector 1", {5, 7}, 8, {{{3, 4}, {2, 3}}}, 0);
  testSubtilesDisjoint<Type, Device::CPU>("Test Vector 2", {5, 7}, 8,
                                          {{{4, 3}, {0, 0}}, {{4, 6}, {1, 1}}}, 1);
  testSubtilesDisjoint<Type, Device::CPU>("Test Vector 3", {5, 7}, 8,
                                          {{{5, 7}, {0, 0}},
                                           {{1, 2}, {2, 2}},
                                           {{3, 0}, {2, 7}},
                                           {{0, 0}, {1, 4}},
                                           {{0, 4}, {3, 3}}},
                                          2);
  testSubtilesDisjoint<Type, Device::CPU>("Test Vector 4", {5, 7}, 8,
                                          {{{5, 7}, {0, 0}}, {{5, 4}, {0, 3}}, {{2, 7}, {3, 0}}}, 1);

  testSubOfSubtile<Type, Device::CPU>("Test SubSub 1", {6, 7}, 6, {{{2, 3}, {3, 3}}, {{4, 6}, {1, 1}}},
                                      {{0, 2}, {2, 1}});
}
