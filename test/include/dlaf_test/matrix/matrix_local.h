//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2020, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <type_traits>

#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/tile.h"
#include "dlaf/types.h"
#include "dlaf/util_math.h"

namespace dlaf {
namespace matrix {
namespace test {

template <class T>
struct MatrixLocal;

/// MatrixLocal is a local matrix with column-layout, not thread-safe
///
/// It is a useful helper object that allows you to access element directly given their
/// GlobalElementIndex, or by accessing the tile via its GlobalTileIndex and then the
/// related LocalTileIndex inside it.
///
/// It uses Index/Size with the Global tag instead of the Local one, because its main task
/// is to create a local copy of a distributed matrix. So, it is generally easier to think
/// of it as the global matrix.
template <class T>
struct MatrixLocal<const T> {
  using ConstTileT = Tile<const T, Device::CPU>;

  MatrixLocal(GlobalElementSize sz, TileElementSize blocksize) noexcept
      : layout_(colMajorLayout({sz.rows(), sz.cols()}, blocksize, sz.rows())), memory_{
                                                                                   layout_.minMemSize()} {
    using dlaf::util::size_t::mul;

    for (const auto& tile_index : iterate_range2d(layout_.nrTiles())) {
      memory::MemoryView<T, Device::CPU> tile_memory{memory_, layout_.tileOffset(tile_index),
                                                     layout_.minTileMemSize(tile_index)};
      tiles_.emplace_back(layout_.tileSize(tile_index), std::move(tile_memory), layout_.ldTile());
    }

    DLAF_ASSERT_HEAVY(ld() == sz.rows(), ld(), sz.rows());
    DLAF_ASSERT_HEAVY(size() == sz, size(), sz);
    DLAF_ASSERT_HEAVY(tiles_.size() == mul(layout_.nrTiles().rows(), layout_.nrTiles().cols()),
                      tiles_.size());
  }

  /// Access directly elements
  const T* ptr(const GlobalElementIndex& index = {0, 0}) const noexcept {
    return memory_(elementLinearIndex(index));
  }

  /// Access a tile
  const ConstTileT& operator()(const GlobalTileIndex& index) const noexcept {
    return tiles_[tileLinearIndex(index)];
  }

  SizeType ld() const noexcept {
    return layout_.size().rows();
  }

  GlobalElementSize size() const noexcept {
    return GlobalElementSize{layout_.size().rows(), layout_.size().cols()};
  }

  TileElementSize blockSize() const noexcept {
    return layout_.blockSize();
  }

  GlobalTileSize nrTiles() const noexcept {
    return GlobalTileSize{layout_.nrTiles().rows(), layout_.nrTiles().cols()};
  }

  friend std::ostream& operator<<(std::ostream& os, const MatrixLocal& matrix) {
    using dlaf::util::size_t::mul;
    using dlaf::util::size_t::sum;

    os << matrix.size() << matrix.ld() << std::endl;
    for (const auto& index : iterate_range2d(matrix.size())) {
      auto linear_index = dlaf::to_sizet(sum(mul(index.col(), matrix.ld()), index.row()));
      os << "[" << index.row() << ", " << index.col() << "]" << '=' << *matrix.memory_(linear_index)
         << "\n";
    }
    return os;
  }

protected:
  std::size_t elementLinearIndex(const GlobalElementIndex& index) const noexcept {
    using dlaf::util::size_t::mul;
    using dlaf::util::size_t::sum;

    DLAF_ASSERT(GlobalElementIndex(index.row(), index.col()).isIn(size()), index, size());

    return sum(mul(index.col(), layout_.ldTile()), index.row());
  }

  SizeType tileLinearIndex(const GlobalTileIndex& index) const noexcept {
    using dlaf::util::size_t::mul;
    using dlaf::util::size_t::sum;

    DLAF_ASSERT(LocalTileIndex(index.row(), index.col()).isIn(layout_.nrTiles()), index,
                layout_.nrTiles());

    return dlaf::to_SizeType(sum(mul(index.col(), layout_.nrTiles().rows()), index.row()));
  }

  const dlaf::matrix::LayoutInfo layout_;
  memory::MemoryView<T, Device::CPU> memory_;
  common::internal::vector<Tile<T, Device::CPU>> tiles_;
};

// Note:
// this is the same workaround used for dlaf::matrix::Matrix in order to be able
// assigning a non-const to a const matrix.
template <class T>
struct MatrixLocal : public MatrixLocal<const T> {
  using TileT = Tile<T, Device::CPU>;

  MatrixLocal(GlobalElementSize size, TileElementSize blocksize) noexcept
      : MatrixLocal<const T>(size, blocksize) {}

  /// Access an element
  T* ptr(const GlobalElementIndex& index = {0, 0}) const noexcept {
    return memory_(elementLinearIndex(index));
  }

  /// Access a tile
  const TileT& operator()(const GlobalTileIndex& index) const noexcept {
    return tiles_[tileLinearIndex(index)];
  }

protected:
  using MatrixLocal<const T>::elementLinearIndex;
  using MatrixLocal<const T>::tileLinearIndex;
  using MatrixLocal<const T>::layout_;
  using MatrixLocal<const T>::memory_;
  using MatrixLocal<const T>::tiles_;
};

}
}
}
