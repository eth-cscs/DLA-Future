//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once
#include <vector>
#include "dlaf/matrix/layout_info.h"
#include "dlaf/matrix_base.h"
#include "dlaf/tile.h"
#include "dlaf/types.h"

namespace dlaf {

template <class T, Device device>
class Matrix : public MatrixBase {
public:
  using ElementType = T;
  using TileType = Tile<ElementType, device>;
  using ConstTileType = Tile<const ElementType, device>;
  friend Matrix<const ElementType, device>;

  Matrix(const GlobalElementSize& size, const TileElementSize& block_size);

  Matrix(const matrix::LayoutInfo& layout, ElementType* ptr, std::size_t elements);

  Matrix(const Matrix& rhs) = delete;
  Matrix(Matrix&& rhs) = default;

  Matrix& operator=(const Matrix& rhs) = delete;
  Matrix& operator=(Matrix&& rhs) = default;

  /// @brief Returns a read-only shared_future of index Tile.
  /// TODO: Sync details.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(nrTiles()) == true.
  hpx::shared_future<ConstTileType> read(const LocalTileIndex& index);

  /// @brief Returns a future of index Tile.
  /// TODO: Sync details.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(nrTiles()) == true.
  hpx::future<TileType> operator()(const LocalTileIndex& index);

protected:
  /// @brief Returns the position in the vector of the index Tile.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(nrTiles()) == true.
  std::size_t tileLinearIndex(const LocalTileIndex& index) {
    assert(index.isValid() && index.isIn(nr_tiles_));
    using util::size_t::sum;
    using util::size_t::mul;
    return sum(index.row(), mul(ld_futures_, index.col()));
  }

private:
  void setUpTiles(const memory::MemoryView<ElementType, device>& mem, const matrix::LayoutInfo& layout);

  std::vector<hpx::future<TileType>> tile_futures_;
  std::vector<hpx::shared_future<ConstTileType>> tile_shared_futures_;
  std::size_t ld_futures_;
};

#include <dlaf/matrix.ipp>

template <class T, Device device>
class Matrix<const T, device> : public MatrixBase {
public:
  using ElementType = T;
  using ConstTileType = Tile<const ElementType, device>;

  Matrix(const matrix::LayoutInfo& layout, ElementType* ptr, std::size_t elements);

  Matrix(const matrix::LayoutInfo& layout, const ElementType* ptr, std::size_t elements)
      : Matrix(layout, const_cast<ElementType*>(ptr), elements) {}

  Matrix(const Matrix& rhs) = delete;
  Matrix(Matrix&& rhs) = default;

  Matrix& operator=(const Matrix& rhs) = delete;
  Matrix& operator=(Matrix&& rhs) = default;

  /// @brief Returns a read-only shared_future of index Tile.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(nrTiles()) == true.
  hpx::shared_future<ConstTileType> read(const LocalTileIndex& index) noexcept;

protected:
  /// @brief Returns the position in the vector of the index Tile.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(nrTiles()) == true.
  std::size_t tileLinearIndex(const LocalTileIndex& index) noexcept {
    assert(index.isValid() && index.isIn(nr_tiles_));
    using util::size_t::sum;
    using util::size_t::mul;
    return sum(index.row(), mul(ld_futures_, index.col()));
  }

private:
  void setUpTiles(const memory::MemoryView<ElementType, device>& mem, const matrix::LayoutInfo& layout);

  std::vector<hpx::shared_future<ConstTileType>> tile_shared_futures_;
  std::size_t ld_futures_;
};

#include <dlaf/matrix_const.ipp>

template <class T, Device device>
Matrix<T, device> createMatrixFromColMajor(const GlobalElementSize& size,
                                           const TileElementSize& block_size, SizeType ld, T* ptr,
                                           std::size_t elements) {
  return Matrix<T, device>(matrix::colMajorLayout(size, block_size, ld), ptr, elements);
}

template <class T, Device device>
Matrix<T, device> createMatrixFromTile(const GlobalElementSize& size, const TileElementSize& block_size,
                                       T* ptr, std::size_t elements) {
  return Matrix<T, device>(matrix::tileLayout(size, block_size), ptr, elements);
}

template <class T, Device device>
Matrix<T, device> createMatrixFromTile(const GlobalElementSize& size, const TileElementSize& block_size,
                                       SizeType ld_tile, SizeType tiles_per_col, T* ptr,
                                       std::size_t elements) {
  return Matrix<T, device>(matrix::tileLayout(size, block_size, ld_tile, tiles_per_col), ptr, elements);
}
}
