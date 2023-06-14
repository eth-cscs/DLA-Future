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
  // TODO: This may need to pass tile size and block size
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

}
}
