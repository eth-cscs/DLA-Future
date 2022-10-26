//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

namespace dlaf {
namespace matrix {

template <class T, Device D>
Matrix<T, D>::Matrix(const LocalElementSize& size, const TileElementSize& block_size)
    : Matrix<T, D>(Distribution(size, block_size)) {}

template <class T, Device D>
Matrix<T, D>::Matrix(const GlobalElementSize& size, const TileElementSize& block_size,
                     const comm::CommunicatorGrid& comm)
    : Matrix<T, D>(Distribution(size, block_size, comm.size(), comm.rank(), {0, 0})) {}

template <class T, Device D>
Matrix<T, D>::Matrix(Distribution distribution) : Matrix<const T, D>(std::move(distribution)) {
  const SizeType alignment = 64;
  const SizeType ld =
      std::max<SizeType>(1,
                         util::ceilDiv(this->distribution().localSize().rows(), alignment) * alignment);

  auto layout = colMajorLayout(this->distribution().localSize(), this->blockSize(), ld);

  SizeType memory_size = layout.minMemSize();
  memory::MemoryView<ElementType, D> mem(memory_size);

  setUpTiles(mem, layout);
}

template <class T, Device D>
Matrix<T, D>::Matrix(Distribution distribution, const LayoutInfo& layout) noexcept
    : Matrix<const T, D>(std::move(distribution)) {
  DLAF_ASSERT(this->distribution().localSize() == layout.size(),
              "Size of distribution does not match layout size!", distribution.localSize(),
              layout.size());
  DLAF_ASSERT(this->distribution().blockSize() == layout.blockSize(), distribution.blockSize(),
              layout.blockSize());

  memory::MemoryView<ElementType, D> mem(layout.minMemSize());

  setUpTiles(mem, layout);
}

template <class T, Device D>
Matrix<T, D>::Matrix(Distribution distribution, const LayoutInfo& layout, ElementType* ptr) noexcept
    : Matrix<const T, D>(std::move(distribution), layout, ptr) {}

template <class T, Device D>
Matrix<T, D>::Matrix(const LayoutInfo& layout, ElementType* ptr) : Matrix<const T, D>(layout, ptr) {}

template <class T, Device D>
pika::future<Tile<T, D>> Matrix<T, D>::operator()(const LocalTileIndex& index) noexcept {
  const auto i = tileLinearIndex(index);
  return tile_managers_[i].getRWTileFuture();
}

}
}
