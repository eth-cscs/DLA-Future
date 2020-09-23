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
    : Matrix<T, device>(matrix::Distribution(size, block_size)) {}

template <class T, Device device>
Matrix<T, device>::Matrix(const GlobalElementSize& size, const TileElementSize& block_size,
                          const comm::CommunicatorGrid& comm)
    : Matrix<T, device>(matrix::Distribution(size, block_size, comm.size(), comm.rank(), {0, 0})) {}

template <class T, Device device>
Matrix<T, device>::Matrix(matrix::Distribution&& distribution)
    : Matrix<const T, device>(std::move(distribution), {}, {}) {
  const SizeType alignment = 64;
  const SizeType ld =
      std::max<SizeType>(1,
                         util::ceilDiv(this->distribution().localSize().rows(), alignment) * alignment);

  auto layout = matrix::colMajorLayout(this->distribution().localSize(), this->blockSize(), ld);

  std::size_t memory_size = layout.minMemSize();
  memory::MemoryView<ElementType, device> mem(memory_size);

  setUpTiles(mem, layout);
}

template <class T, Device device>
Matrix<T, device>::Matrix(matrix::Distribution&& distribution, const matrix::LayoutInfo& layout) noexcept
    : Matrix<const T, device>(std::move(distribution), {}, {}) {
  DLAF_ASSERT(this->distribution().localSize() == layout.size(),
              "Size of distribution does not match layout size!", distribution.localSize(),
              layout.size());
  DLAF_ASSERT(this->distribution().blockSize() == layout.blockSize(),
              "Block size of distribution does not match layout block size!", distribution.blockSize(),
              layout.blockSize());

  memory::MemoryView<ElementType, device> mem(layout.minMemSize());

  setUpTiles(mem, layout);
}

template <class T, Device device>
Matrix<T, device>::Matrix(matrix::Distribution&& distribution, const matrix::LayoutInfo& layout,
                          ElementType* ptr) noexcept
    : Matrix<const T, device>(std::move(distribution), layout, ptr) {}

template <class T, Device device>
Matrix<T, device>::Matrix(const matrix::LayoutInfo& layout, ElementType* ptr)
    : Matrix<const T, device>(layout, ptr) {}

template <class T, Device device>
hpx::future<Tile<T, device>> Matrix<T, device>::operator()(const LocalTileIndex& index) noexcept {
  std::size_t i = tileLinearIndex(index);
  hpx::future<TileType> old_future = std::move(tile_futures_[i]);
  hpx::lcos::local::promise<TileType> p;
  tile_futures_[i] = p.get_future();
  tile_shared_futures_[i] = {};
  return old_future.then(hpx::launch::sync, [p = std::move(p)](hpx::future<TileType>&& fut) mutable {
    try {
      return std::move(fut.get().setPromise(std::move(p)));
    }
    catch (...) {
      auto current_exception_ptr = std::current_exception();
      p.set_exception(current_exception_ptr);
      std::rethrow_exception(current_exception_ptr);
    }
  });
}
