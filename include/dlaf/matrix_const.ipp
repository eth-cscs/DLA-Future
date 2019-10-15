//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

template <class T, Device device>
Matrix<const T, device>::Matrix(const matrix::LayoutInfo& layout, ElementType* ptr, std::size_t elements)
    : MatrixBase(layout.size(), layout.blockSize()) {
  std::size_t memory_size = layout.minMemSize();
  if (elements < memory_size) {
    throw std::invalid_argument("Error: Cannot build Matrix. The memory is too small.");
  }

  memory::MemoryView<ElementType, device> mem(ptr, elements);

  setUpTiles(mem, layout);
}

template <class T, Device device>
hpx::shared_future<Tile<const T, device>> Matrix<const T, device>::read(
    const LocalTileIndex& index) noexcept {
  std::size_t i = tileLinearIndex(index);
  return tile_shared_futures_[i];
}

template <class T, Device device>
void Matrix<const T, device>::setUpTiles(const memory::MemoryView<T, device>& mem,
                                         const matrix::LayoutInfo& layout) {
  using util::size_t::mul;
  const auto& nr_tiles = layout.nrTiles();
  ld_futures_ = static_cast<std::size_t>(nr_tiles.rows());
  std::size_t tot_tiles = mul(ld_futures_, nr_tiles.cols());
  tile_shared_futures_.reserve(tot_tiles);

  using MemView = memory::MemoryView<T, device>;
  for (SizeType j = 0; j < nr_tiles.cols(); ++j) {
    for (SizeType i = 0; i < nr_tiles.rows(); ++i) {
      LocalTileIndex ind(i, j);
      TileElementSize tile_size = layout.tileSize(ind);
      tile_shared_futures_.emplace_back(hpx::make_ready_future<ConstTileType>(
          ConstTileType(tile_size,
                        MemView(mem, layout.tileOffset(ind), layout.minTileMemSize(tile_size)),
                        layout.ldTile())));
    }
  }
}
