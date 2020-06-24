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
Matrix<const T, device>::Matrix(const matrix::LayoutInfo& layout, ElementType* ptr)
    : MatrixBase({layout.size(), layout.blockSize()}) {
  memory::MemoryView<ElementType, device> mem(ptr, layout.minMemSize());
  setUpTiles(mem, layout);
}

template <class T, Device device>
Matrix<const T, device>::Matrix(matrix::Distribution&& distribution, const matrix::LayoutInfo& layout,
                                ElementType* ptr) noexcept
    : MatrixBase(std::move(distribution)) {
  DLAF_ASSERT(this->distribution().localSize() == layout.size(),
              "Size of distribution does not match layout size!", distribution.localSize(),
              layout.size());
  DLAF_ASSERT(this->distribution().blockSize() == layout.blockSize(),
              "Block size of distribution does not match layout block size!", distribution.blockSize(),
              layout.blockSize());

  memory::MemoryView<ElementType, device> mem(ptr, layout.minMemSize());
  setUpTiles(mem, layout);
}

template <class T, Device device>
Matrix<const T, device>::~Matrix() {
  tile_shared_futures_.clear();

  for (auto&& tile_future : tile_futures_) {
    try {
      tile_future.get();
    }
    catch (...) {
      // TODO WARNING
    }
  }
}

template <class T, Device device>
hpx::shared_future<Tile<const T, device>> Matrix<const T, device>::read(
    const LocalTileIndex& index) noexcept {
  std::size_t i = tileLinearIndex(index);
  if (!tile_shared_futures_[i].valid()) {
    hpx::future<TileType> old_future = std::move(tile_futures_[i]);
    hpx::lcos::local::promise<TileType> p;
    tile_futures_[i] = p.get_future();
    tile_shared_futures_[i] = std::move(
        old_future.then(hpx::launch::sync, [p = std::move(p)](hpx::future<TileType>&& fut) mutable {
          try {
            return ConstTileType(std::move(fut.get().setPromise(std::move(p))));
          }
          catch (...) {
            auto current_exception_ptr = std::current_exception();
            p.set_exception(current_exception_ptr);
            std::rethrow_exception(current_exception_ptr);
          }
        }));
  }
  return tile_shared_futures_[i];
}

template <class T, Device device>
Matrix<const T, device>::Matrix(matrix::Distribution&& distribution,
                                std::vector<hpx::future<TileType>>&& tile_futures,
                                std::vector<hpx::shared_future<ConstTileType>>&& tile_shared_futures)
    : MatrixBase(std::move(distribution)), tile_futures_(std::move(tile_futures)),
      tile_shared_futures_(std::move(tile_shared_futures)) {}

template <class T, Device device>
void Matrix<const T, device>::setUpTiles(const memory::MemoryView<ElementType, device>& mem,
                                         const matrix::LayoutInfo& layout) noexcept {
  const auto& nr_tiles = layout.nrTiles();
  tile_shared_futures_.resize(futureVectorSize(nr_tiles));

  tile_futures_.clear();
  tile_futures_.reserve(futureVectorSize(nr_tiles));

  using MemView = memory::MemoryView<T, device>;

  for (SizeType j = 0; j < nr_tiles.cols(); ++j) {
    for (SizeType i = 0; i < nr_tiles.rows(); ++i) {
      LocalTileIndex ind(i, j);
      TileElementSize tile_size = layout.tileSize(ind);
      tile_futures_.emplace_back(hpx::make_ready_future(
          TileType(tile_size, MemView(mem, layout.tileOffset(ind), layout.minTileMemSize(tile_size)),
                   layout.ldTile())));
    }
  }
}
