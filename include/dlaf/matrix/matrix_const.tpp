//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
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
  DLAF_ASSERT(this->distribution().blockSize() == layout.blockSize(), distribution.blockSize(),
              layout.blockSize());

  memory::MemoryView<ElementType, D> mem(ptr, layout.minMemSize());
  setUpTiles(mem, layout);
}

template <class T, Device D>
pika::shared_future<Tile<const T, D>> Matrix<const T, D>::read(const LocalTileIndex& index) noexcept {
  DLAF_UNREACHABLE_PLAIN;
  const auto i = tileLinearIndex(index);
  return tile_managers_[i].getReadTileSharedFuture();
}

template <class T, Device D>
void Matrix<const T, D>::waitLocalTiles() noexcept {
  // Note:
  // Using a readwrite access to the tile ensures that the access is exclusive and not shared
  // among multiple tasks.

  auto readwrite_f = [this](const LocalTileIndex& index) {
    const auto i = tileLinearIndex(index);
    return this->tile_managers_[i].getRWTileFuture();
  };

  const auto range_local = common::iterate_range2d(distribution().localNrTiles());
  pika::wait_all(internal::selectGeneric(readwrite_f, range_local));

  // TODO: This is temporary. Eventually only the below should be used and the
  // above should be deleted.
  auto s = pika::execution::experimental::when_all_vector(internal::selectGeneric(
               [this](const LocalTileIndex& index) {
                 return this->tile_managers_senders_[tileLinearIndex(index)].readwrite();
               },
               range_local)) |
           pika::execution::experimental::drop_value();
  pika::this_thread::experimental::sync_wait(std::move(s));
}

template <class T, Device D>
void Matrix<const T, D>::setUpTiles(const memory::MemoryView<ElementType, D>& mem,
                                    const LayoutInfo& layout) noexcept {
  const auto& nr_tiles = layout.nrTiles();

  tile_managers_.clear();
  tile_managers_.reserve(futureVectorSize(nr_tiles));

  using MemView = memory::MemoryView<T, D>;

  for (SizeType j = 0; j < nr_tiles.cols(); ++j) {
    for (SizeType i = 0; i < nr_tiles.rows(); ++i) {
      LocalTileIndex ind(i, j);
      TileElementSize tile_size = layout.tileSize(ind);
      tile_managers_.emplace_back(
          TileDataType(tile_size, MemView(mem, layout.tileOffset(ind), layout.minTileMemSize(tile_size)),
                       layout.ldTile()));
      tile_managers_senders_.emplace_back(
          TileDataType(tile_size, MemView(mem, layout.tileOffset(ind), layout.minTileMemSize(tile_size)),
                       layout.ldTile()));
    }
  }
}

}
}
