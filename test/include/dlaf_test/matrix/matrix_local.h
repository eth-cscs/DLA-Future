//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cstddef>
#include <type_traits>
#include <vector>

#include <dlaf/common/range2d.h>
#include <dlaf/matrix/col_major_layout.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/memory/memory_view.h>
#include <dlaf/types.h>
#include <dlaf/util_math.h>

namespace dlaf {
namespace matrix {
namespace test {

template <class T>
struct MatrixLocal;

/// MatrixLocal is a column-major local matrix, not thread-safe.
///
/// It is a useful helper object that allows you to access elements directly given their
/// GlobalElementIndex, or by accessing the tile containing it via its GlobalTileIndex and then the
/// related LocalTileIndex inside it.
///
/// Thanks to operator()(GlobalTileIndex) it can be used as a function, which turns out to be useful for
/// comparisons with CHECK macros.
///
/// The column-major layout allow to use it with BLAS/LAPACK routines.
///
/// It is neither thread-safe nor async-ready (see dlaf::Matrix for that).
///
/// It uses Index/Size with the Global tag instead of the Local one, because its main task
/// is to create a local copy of a distributed matrix. So, it is generally easier to think
/// of it as the global matrix.
///
/// About its semantic, it is similar to a std::unique_ptr.
template <class T>
struct MatrixLocal<const T> : public ::dlaf::matrix::internal::MatrixBase {
  using MemoryT = memory::MemoryView<T, Device::CPU>;
  using ConstTileT = Tile<const T, Device::CPU>;

  /// Create a matrix with given size and blocksize
  //
  /// @pre !sz.isEmpty()
  /// @pre !blocksize.isEmpty()
  MatrixLocal(GlobalElementSize sz, TileElementSize tile_size) noexcept
      : MatrixBase(Distribution{sz, tile_size, {1, 1}, {0, 0}, {0, 0}}),
        ld_(std::max<SizeType>(1, sz.rows())) {
    if (sz.isEmpty())
      return;

    ColMajorLayout layout(distribution(), ld_);
    memory_ = MemoryT{layout.min_mem_size()};

    for (const auto& tile_index : iterate_range2d(layout.nr_tiles()))
      tiles_.emplace_back(
          layout.tile_size_of(tile_index),
          MemoryT{memory_, layout.tile_offset(tile_index), layout.min_tile_mem_size(tile_index)}, ld_);
  }

  MatrixLocal(const MatrixLocal&) = delete;
  MatrixLocal& operator=(const MatrixLocal&) = delete;

  MatrixLocal(MatrixLocal&&) = default;

  /// Access elements
  const T* ptr(const GlobalElementIndex& index = {0, 0}) const noexcept {
    return memory_(element_linear_index(index));
  }

  /// Access elements
  const T& operator()(const GlobalElementIndex& index) const noexcept {
    return *ptr(index);
  }

  /// Access tiles
  const ConstTileT& tile_read(const GlobalTileIndex& index) const noexcept {
    return tiles_[tile_linear_index(LocalTileIndex{index.row(), index.col()})];
  }

  SizeType ld() const noexcept {
    return ld_;
  }

protected:
  SizeType element_linear_index(const GlobalElementIndex& index) const noexcept {
    DLAF_ASSERT(index.isIn(size()), index, size());

    return index.row() + index.col() * ld_;
  }

  SizeType ld_;
  MemoryT memory_;

  // Note: this is non-const so that it can be used also by the inheriting class
  std::vector<Tile<T, Device::CPU>> tiles_;
};

// Note:
// this is the same workaround used for dlaf::matrix::Matrix in order to be able
// assigning a non-const to a const matrix.
template <class T>
struct MatrixLocal : public MatrixLocal<const T> {
  using TileT = Tile<T, Device::CPU>;

  MatrixLocal(GlobalElementSize size, TileElementSize blocksize) noexcept
      : MatrixLocal<const T>(size, blocksize) {}

  MatrixLocal(const MatrixLocal&) = delete;
  MatrixLocal& operator=(const MatrixLocal&) = delete;

  MatrixLocal(MatrixLocal&&) = default;

  /// Access elements
  T* ptr(const GlobalElementIndex& index = {0, 0}) const noexcept {
    return memory_(element_linear_index(index));
  }

  /// Access elements
  T& operator()(const GlobalElementIndex& index) const noexcept {
    return *ptr(index);
  }

  /// Access tiles
  const TileT& tile(const GlobalTileIndex& index) const noexcept {
    return tiles_[tile_linear_index(LocalTileIndex{index.row(), index.col()})];
  }

protected:
  using BaseT = MatrixLocal<const T>;
  using BaseT::element_linear_index;
  using BaseT::memory_;
  using BaseT::tile_linear_index;
  using BaseT::tiles_;
};

}
}
}
