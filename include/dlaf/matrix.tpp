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
Matrix<T, device>::Matrix(const LocalElementSize& size, const TileElementSize& block_size)
    : Matrix<const T, device>(matrix::Distribution(size, block_size), {}, {}) {
  SizeType ld = std::max(1, util::ceilDiv(this->size().rows(), 64) * 64);

  auto layout = matrix::colMajorLayout(LocalElementSize(this->size().rows(), this->size().cols()),
                                       this->blockSize(), ld);

  std::size_t memory_size = layout.minMemSize();
  memory::MemoryView<ElementType, device> mem(memory_size);

  setUpTiles(mem, layout);
}

template <class T, Device device>
Matrix<T, device>::Matrix(const matrix::LayoutInfo& layout, ElementType* ptr, std::size_t elements)
    : Matrix<const T, device>(matrix::Distribution(layout.size(), layout.blockSize()), {}, {}) {
  std::size_t memory_size = layout.minMemSize();
  if (elements < memory_size) {
    throw std::invalid_argument("Error: Cannot build Matrix. The memory is too small.");
  }

  memory::MemoryView<ElementType, device> mem(ptr, elements);

  setUpTiles(mem, layout);
}

template <class T, Device device>
hpx::future<Tile<T, device>> Matrix<T, device>::operator()(const LocalTileIndex& index) noexcept {
  std::size_t i = tileLinearIndex(index);
  hpx::future<TileType> old_future = std::move(tile_futures_[i]);
  hpx::promise<TileType> p;
  tile_futures_[i] = p.get_future();
  tile_shared_futures_[i] = {};
  return old_future.then(hpx::launch::sync, [p = std::move(p)](hpx::future<TileType>&& fut) mutable {
    return std::move(fut.get().setPromise(std::move(p)));
  });
}
