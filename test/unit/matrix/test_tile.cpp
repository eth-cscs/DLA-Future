//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <pika/execution.hpp>

#include <dlaf/matrix/index.h>
#include <dlaf/matrix/internal/tile_pipeline.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/memory/memory_view.h>

#include <gtest/gtest.h>

#include <dlaf_test/matrix/util_tile.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;
using dlaf::common::internal::unwrap;

using pika::execution::experimental::then;
using pika::this_thread::experimental::sync_wait;

const std::vector<SizeType> sizes({0, 1, 13, 32});
constexpr SizeType m = 37;
constexpr SizeType n = 87;
constexpr SizeType ld = 133;

SizeType elIndex(TileElementIndex index, SizeType ld) {
  return index.row() + ld * index.col();
}

using TileSizes = std::tuple<TileElementSize, SizeType>;

template <class T, Device D>
TileSizes getSizes(const Tile<T, D>& tile) {
  return TileSizes(tile.size(), tile.ld());
}

TEST(TilePipeline, ResetValid) {
  // The pipeline is valid after construction
  dlaf::matrix::internal::TilePipeline<float, Device::CPU> pipeline({});
  ASSERT_TRUE(pipeline.valid());

  // The pipeline can be reset and is invalid afterwards
  pipeline.reset();
  ASSERT_FALSE(pipeline.valid());

  // The pipeline can be reset multiple times and remains invalid
  pipeline.reset();
  ASSERT_FALSE(pipeline.valid());
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

TYPED_TEST(TileTest, MoveAssignment) {
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

TYPED_TEST(TileTest, MoveAssignmentConst) {
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

TYPED_TEST(TileTest, MoveAssignmentMix) {
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
  typename Tile<T, D>::TileDataType tile(size, memory_view, ld);
  Tile<T, D> tile2(size, std::move(memory_view), ld);
  auto tile_ptr = [tile2 = std::move(tile2)](const TileElementIndex& index) { return tile2.ptr(index); };
  return std::make_tuple(std::move(tile), std::move(tile_ptr));
}

template <class T, Device D>
dlaf::matrix::internal::TilePipeline<T, D> createTilePipeline(Tile<T, D>&& tile) {
  return dlaf::matrix::internal::TilePipeline<T, D>(std::move(tile));
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

template <class SenderWrapper>
void checkNonReady(const std::vector<SenderWrapper>& subtiles) {
  for (const auto& subtile : subtiles) {
    EXPECT_FALSE(subtile.is_ready());
  }
}

template <class T, Device D>
void testShareReadWriteTile(std::string name, TileElementSize size, SizeType ld) {
  SCOPED_TRACE(name);

  auto [tile, tile_ptr] = createTileAndPtrChecker<T, D>(size, ld);
  auto pipeline = createTilePipeline<T, D>(std::move(tile));

  EagerReadWriteTileSender<T, D> first_tile(pipeline.readwrite());
  EagerReadOnlyTileSender<T, D> second_tile1 = shareReadWriteTile(pipeline.readwrite());
  EagerReadOnlyTileSender<T, D> second_tile2 = second_tile1;
  EagerReadWriteTileSender<T, D> third_tile(pipeline.readwrite());

  ASSERT_TRUE(first_tile.is_ready());
  ASSERT_FALSE(second_tile1.is_ready());
  ASSERT_FALSE(second_tile2.is_ready());
  ASSERT_FALSE(third_tile.is_ready());
  checkFullTile(tile_ptr, std::move(first_tile).get(), size);

  ASSERT_TRUE(second_tile1.is_ready());
  ASSERT_TRUE(second_tile2.is_ready());
  ASSERT_FALSE(third_tile.is_ready());
  checkFullTile(tile_ptr, std::move(second_tile1).get().get(), size);

  ASSERT_TRUE(second_tile2.is_ready());
  ASSERT_FALSE(third_tile.is_ready());
  checkFullTile(tile_ptr, std::move(second_tile2).get().get(), size);

  ASSERT_TRUE(third_tile.is_ready());
  checkFullTile(tile_ptr, std::move(third_tile).get(), size);
}

TYPED_TEST(TileTest, ShareReadWriteTile) {
  using Type = TypeParam;

  testShareReadWriteTile<Type, Device::CPU>("Test 1", {1, 1}, 8);
  testShareReadWriteTile<Type, Device::CPU>("Test 2", {5, 7}, 8);
  testShareReadWriteTile<Type, Device::CPU>("Test 3", {512, 256}, 512);
}

// TileSender should be an Eager*Sender. NextTileSender should be an
// EagerReadWriteTileSender.
template <class F, class TileSender, class NextTileSender>
void checkReadyAndDependencyChain(F&& tile_ptr, std::vector<TileSender>& subtiles,
                                  const std::vector<SubTileSpec>& specs, std::size_t last_dep,
                                  NextTileSender& next_tile) {
  ASSERT_EQ(subtiles.size(), specs.size());
  ASSERT_GT(subtiles.size(), last_dep);

  for (const auto& subtile : subtiles) {
    EXPECT_TRUE(subtile.is_ready());
  }
  EXPECT_FALSE(next_tile.is_ready());

  // Check pointer of subtiles (all apart from last_dep) and clear them.
  // As one subtile (last_dep) is still alive next_tile is still locked.
  for (std::size_t i = 0; i < subtiles.size(); ++i) {
    if (i != last_dep) {
      checkSubtile(tile_ptr, unwrap(std::move(subtiles[i]).get()), specs[i]);
    }
  }
  EXPECT_TRUE(subtiles[last_dep].is_ready());
  EXPECT_FALSE(next_tile.is_ready());

  // Check pointer of last_dep subtile and clear it.
  // next_tile should be ready.
  {
    std::size_t i = last_dep;
    checkSubtile(tile_ptr, unwrap(std::move(subtiles[i]).get()), specs[i]);
  }
  EXPECT_TRUE(next_tile.is_ready());
}

template <class T, Device D>
void testSubtileConst(std::string name, TileElementSize size, SizeType ld, const SubTileSpec& spec,
                      std::size_t last_dep) {
  SCOPED_TRACE(name);

  auto [tile, tile_ptr] = createTileAndPtrChecker<T, D>(size, ld);
  auto pipeline = createTilePipeline<T, D>(std::move(tile));

  EagerReadWriteTileSender<T, D> first_tile(pipeline.readwrite());
  auto second_tile_orig = pipeline.read();
  EagerReadOnlyTileSender<T, D> second_tile(second_tile_orig);
  EagerReadOnlyTileSender<T, D> subtile(splitTile(second_tile_orig, spec));
  std::vector<EagerReadOnlyTileSender<T, D>> subtiles = {std::move(subtile), std::move(second_tile)};
  std::vector<SubTileSpec> full_specs = {spec, {{0, 0}, size}};
  second_tile_orig = {};
  EagerReadWriteTileSender<T, D> third_tile(pipeline.readwrite());

  ASSERT_TRUE(first_tile.is_ready());
  checkNonReady(subtiles);
  ASSERT_FALSE(third_tile.is_ready());
  checkFullTile(tile_ptr, std::move(first_tile).get(), size);

  ASSERT_FALSE(third_tile.is_ready());
  checkReadyAndDependencyChain(tile_ptr, subtiles, full_specs, last_dep, third_tile);

  ASSERT_TRUE(third_tile.is_ready());
  checkFullTile(tile_ptr, std::move(third_tile).get(), size);
}

template <class T, Device D>
void testSubtilesConst(std::string name, TileElementSize size, SizeType ld,
                       std::vector<SubTileSpec> specs, std::size_t last_dep) {
  SCOPED_TRACE(name);
  ASSERT_LE(last_dep, specs.size());

  auto [tile, tile_ptr] = createTileAndPtrChecker<T, D>(size, ld);
  auto pipeline = createTilePipeline<T, D>(std::move(tile));

  EagerReadWriteTileSender<T, D> first_tile(pipeline.readwrite());
  auto second_tile_orig = pipeline.read();
  EagerReadOnlyTileSender<T, D> second_tile(second_tile_orig);
  auto subtiles_orig = splitTile(second_tile_orig, specs);
  std::vector<EagerReadOnlyTileSender<T, D>> subtiles;
  subtiles.reserve(specs.size());
  for (auto& subtile : subtiles_orig) {
    subtiles.emplace_back(std::move(subtile));
  }
  subtiles.push_back(std::move(second_tile));
  specs.push_back({{0, 0}, size});
  subtiles_orig.clear();
  second_tile_orig = {};
  EagerReadWriteTileSender<T, D> third_tile(pipeline.readwrite());

  ASSERT_TRUE(first_tile.is_ready());
  checkNonReady(subtiles);
  ASSERT_FALSE(third_tile.is_ready());
  checkFullTile(tile_ptr, std::move(first_tile).get(), size);

  ASSERT_FALSE(third_tile.is_ready());
  if (subtiles.size() > 0) {
    checkReadyAndDependencyChain(tile_ptr, subtiles, specs, last_dep, third_tile);
  }

  ASSERT_TRUE(third_tile.is_ready());
  checkFullTile(tile_ptr, std::move(third_tile).get(), size);
}

template <class T, Device D>
void testSubtilesConstShareReadWriteTile(std::string name, TileElementSize size, SizeType ld,
                                         std::vector<SubTileSpec> specs, std::size_t last_dep) {
  SCOPED_TRACE(name);
  ASSERT_LE(last_dep, specs.size());

  auto [tile, tile_ptr] = createTileAndPtrChecker<T, D>(size, ld);
  auto pipeline = createTilePipeline<T, D>(std::move(tile));

  EagerReadWriteTileSender<T, D> first_tile(pipeline.readwrite());
  auto second_tile_orig = shareReadWriteTile(pipeline.readwrite());
  EagerReadOnlyTileSender<T, D> second_tile(second_tile_orig);
  auto subtiles_orig = splitTile(second_tile_orig, specs);
  std::vector<EagerReadOnlyTileSender<T, D>> subtiles;
  subtiles.reserve(specs.size());
  for (auto& subtile : subtiles_orig) {
    subtiles.emplace_back(std::move(subtile));
  }
  subtiles.push_back(std::move(second_tile));
  specs.push_back({{0, 0}, size});
  subtiles_orig.clear();
  second_tile_orig = {};
  EagerReadWriteTileSender<T, D> third_tile(pipeline.readwrite());

  ASSERT_TRUE(first_tile.is_ready());
  checkNonReady(subtiles);
  ASSERT_FALSE(third_tile.is_ready());
  checkFullTile(tile_ptr, std::move(first_tile).get(), size);

  ASSERT_FALSE(third_tile.is_ready());
  if (subtiles.size() > 0) {
    checkReadyAndDependencyChain(tile_ptr, subtiles, specs, last_dep, third_tile);
  }

  ASSERT_TRUE(third_tile.is_ready());
  checkFullTile(tile_ptr, std::move(third_tile).get(), size);
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
  auto pipeline = createTilePipeline<T, D>(std::move(tile));

  EagerReadWriteTileSender<T, D> first_tile(pipeline.readwrite());
  auto second_tile_orig = pipeline.read();
  auto subtiles_orig = splitTile(second_tile_orig, specs);
  EagerReadWriteTileSender<T, D> third_tile(pipeline.readwrite());

  subtiles_orig.emplace_back(splitTile(subtiles_orig[0], subspec));
  subtiles_orig.emplace_back(std::move(second_tile_orig));
  specs.push_back({specs[0].origin + common::sizeFromOrigin(subspec.origin), subspec.size});
  specs.push_back({{0, 0}, size});

  std::vector<EagerReadOnlyTileSender<T, D>> subtiles;
  subtiles.reserve(subtiles_orig.size());
  for (auto& subtile : subtiles_orig) {
    subtiles.emplace_back(std::move(subtile));
  }
  subtiles_orig.clear();

  ASSERT_TRUE(first_tile.is_ready());
  checkNonReady(subtiles);
  ASSERT_FALSE(third_tile.is_ready());
  checkFullTile(tile_ptr, std::move(first_tile).get(), size);

  ASSERT_FALSE(third_tile.is_ready());
  if (subtiles.size() > 0) {
    checkReadyAndDependencyChain(tile_ptr, subtiles, specs, last_dep, third_tile);
  }

  ASSERT_TRUE(third_tile.is_ready());
  checkFullTile(tile_ptr, std::move(third_tile).get(), size);
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

  testSubtilesConstShareReadWriteTile<Type, Device::CPU>("Test Share Vector Empty", {5, 7}, 8, {}, 0);
  testSubtilesConstShareReadWriteTile<Type, Device::CPU>("Test Share Vector 1", {5, 7}, 8,
                                                         {{{3, 4}, {2, 3}}}, 1);
  testSubtilesConstShareReadWriteTile<Type, Device::CPU>("Test Share Vector 2", {5, 7}, 8,
                                                         {{{4, 3}, {0, 0}}, {{4, 6}, {1, 1}}}, 2);
  testSubtilesConstShareReadWriteTile<Type, Device::CPU>("Test Share Vector 3", {5, 7}, 8,
                                                         {{{5, 7}, {0, 0}},
                                                          {{2, 2}, {2, 2}},
                                                          {{3, 0}, {2, 7}},
                                                          {{0, 0}, {5, 7}}},
                                                         2);
  testSubtilesConst<Type, Device::CPU>("Test Share Vector 4", {5, 7}, 8,
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
  auto pipeline = createTilePipeline<T, D>(std::move(tile));

  EagerReadWriteTileSender<T, D> first_tile(pipeline.readwrite());
  EagerReadWriteTileSender<T, D> subtile(splitTile(pipeline.readwrite(), spec));
  std::vector<EagerReadWriteTileSender<T, D>> subtiles;
  subtiles.push_back(std::move(subtile));
  std::vector<SubTileSpec> full_specs = {spec};
  EagerReadWriteTileSender<T, D> third_tile(pipeline.readwrite());

  ASSERT_TRUE(first_tile.is_ready());
  checkNonReady(subtiles);
  ASSERT_FALSE(third_tile.is_ready());
  checkFullTile(tile_ptr, std::move(first_tile).get(), size);

  ASSERT_FALSE(third_tile.is_ready());
  checkReadyAndDependencyChain(tile_ptr, subtiles, full_specs, 0, third_tile);

  ASSERT_TRUE(third_tile.is_ready());
  checkFullTile(tile_ptr, std::move(third_tile).get(), size);
}

template <class T, Device D>
void testSubtilesDisjoint(std::string name, TileElementSize size, SizeType ld,
                          const std::vector<SubTileSpec>& specs, std::size_t last_dep) {
  SCOPED_TRACE(name);
  if (specs.size() > 0) {
    ASSERT_LT(last_dep, specs.size());
  }

  auto [tile, tile_ptr] = createTileAndPtrChecker<T, D>(size, ld);
  auto pipeline = createTilePipeline<T, D>(std::move(tile));

  EagerReadWriteTileSender<T, D> first_tile(pipeline.readwrite());
  auto subtiles_orig = splitTileDisjoint(pipeline.readwrite(), specs);
  std::vector<EagerReadWriteTileSender<T, D>> subtiles;
  subtiles.reserve(specs.size());
  for (auto& subtile : subtiles_orig) {
    subtiles.emplace_back(std::move(subtile));
  }
  subtiles_orig.clear();
  EagerReadWriteTileSender<T, D> third_tile(pipeline.readwrite());

  ASSERT_TRUE(first_tile.is_ready());
  checkNonReady(subtiles);
  ASSERT_FALSE(third_tile.is_ready());
  checkFullTile(tile_ptr, std::move(first_tile).get(), size);

  if (subtiles.size() > 0) {
    ASSERT_FALSE(third_tile.is_ready());
    checkReadyAndDependencyChain(tile_ptr, subtiles, specs, last_dep, third_tile);
  }

  ASSERT_TRUE(third_tile.is_ready());
  checkFullTile(tile_ptr, std::move(third_tile).get(), size);
}

template <class T, Device D>
void testSubOfSubtile(std::string name, TileElementSize size, SizeType ld,
                      std::vector<SubTileSpec> specs, const SubTileSpec& subspec) {
  SCOPED_TRACE(name);
  ASSERT_LE(1, specs.size());  // Need at least a subtile to create a subsubtile
  // last_dep = 0 -> subsubtile

  auto [tile, tile_ptr] = createTileAndPtrChecker<T, D>(size, ld);
  auto pipeline = createTilePipeline<T, D>(std::move(tile));

  EagerReadWriteTileSender<T, D> first_tile(pipeline.readwrite());
  auto subtiles_orig = splitTileDisjoint(pipeline.readwrite(), specs);
  EagerReadWriteTileSender<T, D> third_tile(pipeline.readwrite());

  // create subsubtile
  subtiles_orig[0] = splitTile(std::move(subtiles_orig[0]), subspec);
  specs[0] = {specs[0].origin + common::sizeFromOrigin(subspec.origin), subspec.size};
  std::vector<EagerReadWriteTileSender<T, D>> subtiles;
  subtiles.reserve(specs.size());
  for (auto& subtile : subtiles_orig) {
    subtiles.emplace_back(std::move(subtile));
  }
  subtiles_orig.clear();

  ASSERT_TRUE(first_tile.is_ready());
  checkNonReady(subtiles);
  ASSERT_FALSE(third_tile.is_ready());
  checkFullTile(tile_ptr, std::move(first_tile).get(), size);

  ASSERT_FALSE(third_tile.is_ready());
  if (subtiles.size() > 0) {
    checkReadyAndDependencyChain(tile_ptr, subtiles, specs, 0, third_tile);
  }

  ASSERT_TRUE(third_tile.is_ready());
  checkFullTile(tile_ptr, std::move(third_tile).get(), size);
}

TYPED_TEST(TileTest, Subtile) {
  using Type = TypeParam;

  testSubtile<Type, Device::CPU>("Test 1", {5, 7}, 8, {{3, 4}, {2, 3}});
  testSubtile<Type, Device::CPU>("Test 2", {5, 7}, 8, {{4, 6}, {1, 1}});
  testSubtile<Type, Device::CPU>("Test 3", {5, 7}, 8, {{0, 0}, {5, 7}});

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
