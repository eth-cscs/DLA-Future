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
#include "dlaf/matrix/tile.h"
#include "dlaf/memory/memory_view.h"
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
  static constexpr auto CPU = Device::CPU;

  using Memory_t = memory::MemoryView<T, CPU>;
  using ConstTileT = Tile<const T, CPU>;

  MatrixLocal(GlobalElementSize sz, TileElementSize blocksize) noexcept
      : layout_{colMajorLayout({sz.rows(), sz.cols()}, blocksize, sz.rows())} {
    using dlaf::util::size_t::mul;

    memory_ = Memory_t{layout_.minMemSize()};

    for (const auto& tile_index : iterate_range2d(layout_.nrTiles()))
      tiles_.emplace_back(layout_.tileSize(tile_index),
                          Memory_t{memory_, layout_.tileOffset(tile_index),
                                   layout_.minTileMemSize(tile_index)},
                          layout_.ldTile());
  }

  MatrixLocal(MatrixLocal&&) = default;

  /// Access elements
  const T* ptr(const GlobalElementIndex& index = {0, 0}) const noexcept {
    return memory_(elementLinearIndex(index));
  }

  /// Access elements
  const T& operator()(const GlobalElementIndex& index) const noexcept {
    return *ptr(index);
  }

  /// Access tiles
  const ConstTileT& tile_read(const GlobalTileIndex& index) const noexcept {
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

protected:
  SizeType elementLinearIndex(const GlobalElementIndex& index) const noexcept {
    DLAF_ASSERT(GlobalElementIndex(index.row(), index.col()).isIn(size()), index, size());

    return index.row() + index.col() * layout_.ldTile();
  }

  SizeType tileLinearIndex(const GlobalTileIndex& index) const noexcept {
    DLAF_ASSERT(LocalTileIndex(index.row(), index.col()).isIn(layout_.nrTiles()), index,
                layout_.nrTiles());

    return index.row() + index.col() * layout_.nrTiles().rows();
  }

  dlaf::matrix::LayoutInfo layout_;
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

  MatrixLocal(MatrixLocal&&) = default;

  /// Access elements
  T* ptr(const GlobalElementIndex& index = {0, 0}) const noexcept {
    return memory_(elementLinearIndex(index));
  }

  /// Access elements
  T& operator()(const GlobalElementIndex& index) const noexcept {
    return *ptr(index);
  }

  /// Access tiles
  const TileT& tile(const GlobalTileIndex& index) const noexcept {
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
