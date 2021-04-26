//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

namespace dlaf {
namespace matrix {

template <class T, Device device>
Matrix<const T, device>::Matrix(const LayoutInfo& layout, ElementType* ptr)
    : MatrixBase({layout.size(), layout.blockSize()}) {
  memory::MemoryView<ElementType, device> mem(ptr, layout.minMemSize());
  setUpTiles(mem, layout);
}

template <class T, Device device>
Matrix<const T, device>::Matrix(Distribution distribution, const matrix::LayoutInfo& layout,
                                ElementType* ptr) noexcept
    : MatrixBase(std::move(distribution)) {
  DLAF_ASSERT(this->distribution().localSize() == layout.size(), distribution.localSize(),
              layout.size());
  DLAF_ASSERT(this->distribution().blockSize() == layout.blockSize(), distribution.blockSize(),
              layout.blockSize());

  memory::MemoryView<ElementType, device> mem(ptr, layout.minMemSize());
  setUpTiles(mem, layout);
}

template <class T, Device device>
hpx::shared_future<Tile<const T, device>> Matrix<const T, device>::read(
    const LocalTileIndex& index) noexcept {
  const auto i = tileLinearIndex(index);
  return tile_managers_[i].getReadTileSharedFuture();
}

template <class T, Device device>
void Matrix<const T, device>::waitLocalTiles() noexcept {
  // Note:
  // Using a readwrite access to the tile ensures that the access is exclusive and not shared
  // among multiple tasks.

  auto readwrite_f = [this](const LocalTileIndex& index) {
    const auto i = tileLinearIndex(index);
    return this->tile_managers_[i].getRWTileFuture();
  };

  const auto range_local = common::iterate_range2d(distribution().localNrTiles());
  auto all_local_tiles_rw = internal::selectGeneric(readwrite_f, range_local);
  hpx::wait_all(std::move(all_local_tiles_rw));
}

template <class T, Device device>
void Matrix<const T, device>::setUpTiles(const memory::MemoryView<ElementType, device>& mem,
                                         const LayoutInfo& layout) noexcept {
  const auto& nr_tiles = layout.nrTiles();

  tile_managers_.clear();
  tile_managers_.reserve(futureVectorSize(nr_tiles));

  using MemView = memory::MemoryView<T, device>;

  for (SizeType j = 0; j < nr_tiles.cols(); ++j) {
    for (SizeType i = 0; i < nr_tiles.rows(); ++i) {
      LocalTileIndex ind(i, j);
      TileElementSize tile_size = layout.tileSize(ind);
      tile_managers_.emplace_back(
          TileType(tile_size, MemView(mem, layout.tileOffset(ind), layout.minTileMemSize(tile_size)),
                   layout.ldTile()));
    }
  }
}

}
}
