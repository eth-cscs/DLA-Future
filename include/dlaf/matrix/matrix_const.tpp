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
#include <pika/thread.hpp>

namespace dlaf {
namespace matrix {

template <class T, Device D>
Matrix<const T, D>::Matrix(const LayoutInfo& layout, ElementType* ptr)
    : MatrixBase({layout.size(), layout.blockSize()}) {
  memory::MemoryView<ElementType, D> mem(ptr, layout.minMemSize());
  setUpTiles(mem, layout);
}

template <class T, Device D>
Matrix<const T, D>::Matrix(Distribution distribution, const matrix::LayoutInfo& layout,
                           ElementType* ptr) noexcept
    : MatrixBase(std::move(distribution)) {
  DLAF_ASSERT(this->distribution().localSize() == layout.size(), distribution.localSize(),
              layout.size());
  DLAF_ASSERT(this->distribution().baseTileSize() == layout.blockSize(), distribution.baseTileSize(),
              layout.blockSize());

  memory::MemoryView<ElementType, D> mem(ptr, layout.minMemSize());
  setUpTiles(mem, layout);
}

template <class T, Device D>
void Matrix<const T, D>::waitLocalTiles() noexcept {
  // Note:
  // Using a readwrite access to the tile ensures that the access is exclusive and not shared
  // among multiple tasks.

  const auto range_local = common::iterate_range2d(distribution().localNrTiles());

  auto s = pika::execution::experimental::when_all_vector(internal::selectGeneric(
               [this](const LocalTileIndex& index) {
                 return this->tile_managers_[tileLinearIndex(index)].readwrite();
               },
               range_local)) |
           pika::execution::experimental::drop_value();
  pika::this_thread::experimental::sync_wait(std::move(s));
}

template <class T, Device D>
void Matrix<const T, D>::setUpTiles(const memory::MemoryView<ElementType, D>& mem,
                                    const LayoutInfo& layout) noexcept {
  const auto& nr_tiles = layout.nrTiles();

  DLAF_ASSERT(tile_managers_.empty(), "");
  tile_managers_.reserve(to_sizet(nr_tiles.linear_size()));

  using MemView = memory::MemoryView<T, D>;

  for (SizeType j = 0; j < nr_tiles.cols(); ++j) {
    for (SizeType i = 0; i < nr_tiles.rows(); ++i) {
      LocalTileIndex ind(i, j);
      TileElementSize tile_size = layout.tileSize(ind);
      tile_managers_.emplace_back(
          TileDataType(tile_size, MemView(mem, layout.tileOffset(ind), layout.minTileMemSize(tile_size)),
                       layout.ldTile()));
    }
  }
}

template <class T, Device D>
Matrix<const T, D>::Matrix(Matrix<const T, D>& mat, const SubPipelineTag)
    : MatrixBase(mat.distribution()) {
  setUpSubPipelines(mat);
}

template <class T, Device D>
Matrix<const T, D>::Matrix(Matrix<const T, D>& mat, const LocalTileSize& tiles_per_block)
    : MatrixBase(mat.distribution(), tiles_per_block) {
  setUpRetiledSubPipelines(mat, tiles_per_block);
}

template <class T, Device D>
void Matrix<const T, D>::setUpSubPipelines(Matrix<const T, D>& mat) noexcept {
  namespace ex = pika::execution::experimental;

  // TODO: Optimize read-after-read. This is currently forced to access the base
  // matrix in readwrite mode so that we can move the tile into the
  // sub-pipeline. This is semantically not required and should eventually be
  // optimized.
  tile_managers_.reserve(mat.tile_managers_.size());
  for (auto& tm : mat.tile_managers_) {
    tile_managers_.emplace_back(Tile<T, D>());
    auto s = ex::when_all(tile_managers_.back().readwrite_with_wrapper(), tm.readwrite()) |
             ex::then([](internal::TileAsyncRwMutexReadWriteWrapper<T, D> empty_tile_wrapper,
                         Tile<T, D> tile) { empty_tile_wrapper.get() = std::move(tile); });
    ex::start_detached(std::move(s));
  }
}

template <class T, Device D>
void Matrix<const T, D>::setUpRetiledSubPipelines(Matrix<const T, D>& mat,
                                                  const LocalTileSize& tiles_per_block) noexcept {
  DLAF_ASSERT(mat.blockSize() == mat.baseTileSize(), mat.blockSize(), mat.baseTileSize());

  using common::internal::vector;
  namespace ex = pika::execution::experimental;

  const auto n = to_sizet(distribution().localNrTiles().linear_size());
  tile_managers_.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    tile_managers_.emplace_back(Tile<T, D>());
  }

  const auto tile_size = distribution().baseTileSize();
  vector<SubTileSpec> specs;
  vector<LocalTileIndex> indices;
  specs.reserve(tiles_per_block.linear_size());
  indices.reserve(tiles_per_block.linear_size());

  // TODO: Optimize read-after-read. This is currently forced to access the base matrix in readwrite mode
  // so that we can move the tile into the sub-pipeline. This is semantically not required and should
  // eventually be optimized.
  for (const auto& orig_tile_index : common::iterate_range2d(mat.distribution().localNrTiles())) {
    const auto original_tile_size = mat.tileSize(mat.distribution().globalTileIndex(orig_tile_index));

    for (SizeType j = 0; j < original_tile_size.cols(); j += tile_size.cols())
      for (SizeType i = 0; i < original_tile_size.rows(); i += tile_size.rows()) {
        indices.emplace_back(
            LocalTileIndex{orig_tile_index.row() * tiles_per_block.rows() + i / tile_size.rows(),
                           orig_tile_index.col() * tiles_per_block.cols() + j / tile_size.cols()});
        specs.emplace_back(SubTileSpec{{i, j},
                                       tileSize(distribution().globalTileIndex(indices.back()))});
      }

    auto sub_tiles =
        splitTileDisjoint(mat.tile_managers_[mat.tileLinearIndex(orig_tile_index)].readwrite(), specs);

    DLAF_ASSERT_HEAVY(specs.size() == indices.size(), specs.size(), indices.size());
    for (SizeType j = 0; j < specs.size(); ++j) {
      const auto i = tileLinearIndex(indices[j]);

      // Move subtile to be managed by the tile manager of RetiledMatrix. We
      // use readwrite_with_wrapper to get access to the original tile managed
      // by the underlying async_rw_mutex.
      auto s =
          ex::when_all(tile_managers_[i].readwrite_with_wrapper(), std::move(sub_tiles[to_sizet(j)])) |
          ex::then([](internal::TileAsyncRwMutexReadWriteWrapper<T, D> empty_tile_wrapper,
                      Tile<T, D> sub_tile) { empty_tile_wrapper.get() = std::move(sub_tile); });
      ex::start_detached(std::move(s));
    }

    specs.clear();
    indices.clear();
  }
}

}
}
