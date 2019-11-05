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
class Matrix : public Matrix<const T, device> {
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

  /// @brief Returns a future of index Tile.
  /// TODO: Sync details.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(nrTiles()) == true.
  hpx::future<TileType> operator()(const LocalTileIndex& index) noexcept;

protected:
  using Matrix<const T, device>::tileLinearIndex;

private:
  using Matrix<const T, device>::setUpTiles;
  using Matrix<const T, device>::futureVectorSize;
  using Matrix<const T, device>::tile_futures_;
  using Matrix<const T, device>::tile_shared_futures_;
};

#include "dlaf/matrix.ipp"

template <class T, Device device>
class Matrix<const T, device> : protected MatrixBase {
public:
  using ElementType = T;
  using TileType = Tile<ElementType, device>;
  using ConstTileType = Tile<const ElementType, device>;
  friend Matrix<ElementType, device>;

  Matrix(const matrix::LayoutInfo& layout, ElementType* ptr, std::size_t elements);

  Matrix(const matrix::LayoutInfo& layout, const ElementType* ptr, std::size_t elements)
      : Matrix(layout, const_cast<ElementType*>(ptr), elements) {}

  Matrix(const Matrix& rhs) = delete;
  Matrix(Matrix&& rhs) = default;

  virtual ~Matrix();

  Matrix& operator=(const Matrix& rhs) = delete;
  Matrix& operator=(Matrix&& rhs) = default;

  using MatrixBase::size;
  using MatrixBase::blockSize;
  using MatrixBase::nrTiles;

  /// @brief Returns a read-only shared_future of index Tile.
  /// TODO: Sync details.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(nrTiles()) == true.
  hpx::shared_future<ConstTileType> read(const LocalTileIndex& index) noexcept;

protected:
  /// @brief Returns the position in the vector of the index Tile.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(nrTiles()) == true.
  std::size_t tileLinearIndex(const LocalTileIndex& index) const noexcept {
    assert(index.isValid() && index.isIn(nrTiles()));
    using util::size_t::sum;
    using util::size_t::mul;
    return sum(index.row(), mul(nrTiles().rows(), index.col()));
  }

private:
  Matrix(const GlobalElementSize& size, const TileElementSize& block_size,
         std::vector<hpx::future<TileType>>&& tile_futures,
         std::vector<hpx::shared_future<ConstTileType>>&& tile_shared_futures);

  void setUpTiles(const memory::MemoryView<ElementType, device>& mem,
                  const matrix::LayoutInfo& layout) noexcept;

  std::size_t futureVectorSize(const matrix::LayoutInfo& layout) const noexcept;

  std::vector<hpx::future<TileType>> tile_futures_;
  std::vector<hpx::shared_future<ConstTileType>> tile_shared_futures_;
};

#include "dlaf/matrix_const.ipp"

// Note: the templates of the following helper functions are inverted w.r.t. the Matrix templates
// to allow the user to only specify the device and let the compiler deduce the type T.

template <Device device, class T>
Matrix<T, device> createMatrixFromColMajor(const GlobalElementSize& size,
                                           const TileElementSize& block_size, SizeType ld, T* ptr,
                                           std::size_t elements) {
  return Matrix<T, device>(matrix::colMajorLayout(size, block_size, ld), ptr, elements);
}

template <Device device, class T>
Matrix<T, device> createMatrixFromTile(const GlobalElementSize& size, const TileElementSize& block_size,
                                       T* ptr, std::size_t elements) {
  return Matrix<T, device>(matrix::tileLayout(size, block_size), ptr, elements);
}

template <Device device, class T>
Matrix<T, device> createMatrixFromTile(const GlobalElementSize& size, const TileElementSize& block_size,
                                       SizeType ld_tile, SizeType tiles_per_col, T* ptr,
                                       std::size_t elements) {
  return Matrix<T, device>(matrix::tileLayout(size, block_size, ld_tile, tiles_per_col), ptr, elements);
}
}
