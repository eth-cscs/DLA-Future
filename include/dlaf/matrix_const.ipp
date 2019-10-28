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

  setUpConstTiles(mem, layout);
}

template <class T, Device device>
hpx::shared_future<Tile<const T, device>> Matrix<const T, device>::read(
    const LocalTileIndex& index) noexcept {
  std::size_t i = tileLinearIndex(index);
  if (!tile_shared_futures_[i].valid()) {
    hpx::future<TileType> old_future = std::move(tile_futures_[i]);
    hpx::promise<TileType> p;
    tile_futures_[i] = p.get_future();
    tile_shared_futures_[i] = std::move(
        old_future.then(hpx::launch::sync, [p = std::move(p)](hpx::future<TileType>&& fut) mutable {
          return ConstTileType(std::move(fut.get().setPromise(std::move(p))));
        }));
  }
  return tile_shared_futures_[i];
}

template <class T, Device device>
Matrix<const T, device>::Matrix(const GlobalElementSize& size, const TileElementSize& block_size,
                                std::vector<hpx::future<TileType>>&& tile_futures,
                                std::vector<hpx::shared_future<ConstTileType>>&& tile_shared_futures)
    : MatrixBase(size, block_size), tile_futures_(std::move(tile_futures)),
      tile_shared_futures_(std::move(tile_shared_futures)) {}

template <class T, Device device>
void Matrix<const T, device>::setUpConstTiles(const memory::MemoryView<T, device>& mem,
                                              const matrix::LayoutInfo& layout) noexcept {
  tile_futures_.resize(futureVectorSize(layout));

  setUpTilesInternal(tile_shared_futures_, mem, layout);
}

template <class T, Device device>
template <template <class> class Future, class TileT>
void Matrix<const T, device>::setUpTilesInternal(std::vector<Future<TileT>>& tile_futures_vector,
                                                 const memory::MemoryView<ElementType, device>& mem,
                                                 const matrix::LayoutInfo& layout) noexcept {
  tile_futures_vector.clear();
  tile_futures_vector.reserve(futureVectorSize(layout));

  using MemView = memory::MemoryView<T, device>;
  const auto& nr_tiles = layout.nrTiles();

  for (SizeType j = 0; j < nr_tiles.cols(); ++j) {
    for (SizeType i = 0; i < nr_tiles.rows(); ++i) {
      LocalTileIndex ind(i, j);
      TileElementSize tile_size = layout.tileSize(ind);
      tile_futures_vector.emplace_back(hpx::make_ready_future<TileT>(
          TileT(tile_size, MemView(mem, layout.tileOffset(ind), layout.minTileMemSize(tile_size)),
                layout.ldTile())));
    }
  }
}

template <class T, Device device>
std::size_t Matrix<const T, device>::futureVectorSize(const matrix::LayoutInfo& layout) const noexcept {
  using util::size_t::mul;
  const auto& nr_tiles = layout.nrTiles();
  return mul(nr_tiles.rows(), nr_tiles.cols());
}
