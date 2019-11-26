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
#include <exception>
#include <vector>
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/layout_info.h"
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

  Matrix(const LocalElementSize& size, const TileElementSize& block_size);

  Matrix(const GlobalElementSize& size, const TileElementSize& block_size,
         const comm::CommunicatorGrid& comm);

  Matrix(matrix::Distribution&& distribution);

  Matrix(matrix::Distribution&& distribution, const matrix::LayoutInfo& layout);

  Matrix(const matrix::LayoutInfo& layout, ElementType* ptr);

  Matrix(matrix::Distribution&& distribution, const matrix::LayoutInfo& layout, ElementType* ptr);

  Matrix(const Matrix& rhs) = delete;
  Matrix(Matrix&& rhs) = default;

  Matrix& operator=(const Matrix& rhs) = delete;
  Matrix& operator=(Matrix&& rhs) = default;

  /// @brief Returns a future of the Tile with local index @p index.
  ///
  /// TODO: Sync details.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(distribution().localNrTiles()) == true.
  hpx::future<TileType> operator()(const LocalTileIndex& index) noexcept;

  /// @brief Returns a future of the Tile with global index @p index.
  ///
  /// TODO: Sync details.
  /// @throws std::invalid_argument if the global tile is not stored in the current process.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(globalNrTiles()) == true.
  hpx::future<TileType> operator()(const GlobalTileIndex& index) noexcept;

protected:
  using Matrix<const T, device>::tileLinearIndex;

private:
  using Matrix<const T, device>::setUpTiles;
  using Matrix<const T, device>::futureVectorSize;
  using Matrix<const T, device>::tile_futures_;
  using Matrix<const T, device>::tile_shared_futures_;
};

#include "dlaf/matrix.tpp"

template <class T, Device device>
class Matrix<const T, device> : protected matrix::Distribution {
public:
  using ElementType = T;
  using TileType = Tile<ElementType, device>;
  using ConstTileType = Tile<const ElementType, device>;
  friend Matrix<ElementType, device>;

  Matrix(const matrix::LayoutInfo& layout, ElementType* ptr);

  Matrix(const matrix::LayoutInfo& layout, const ElementType* ptr)
      : Matrix(layout, const_cast<ElementType*>(ptr)) {}

  Matrix(matrix::Distribution&& distribution, const matrix::LayoutInfo& layout, ElementType* ptr);

  Matrix(matrix::Distribution&& distribution, const matrix::LayoutInfo& layout, const ElementType* ptr)
      : Matrix(std::move(distribution), layout, const_cast<ElementType*>(ptr)) {}

  Matrix(const Matrix& rhs) = delete;
  Matrix(Matrix&& rhs) = default;

  virtual ~Matrix();

  Matrix& operator=(const Matrix& rhs) = delete;
  Matrix& operator=(Matrix&& rhs) = default;

  using Distribution::size;
  using Distribution::blockSize;
  using Distribution::nrTiles;

  using Distribution::rankIndex;
  using Distribution::commGridSize;

  using Distribution::rankGlobalTile;

  const matrix::Distribution& distribution() const noexcept {
    return *this;
  }

  /// Returns a read-only shared_future of the Tile with local index @p index.
  ///
  /// TODO: Sync details.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(distribution().localNrTiles()) == true.
  hpx::shared_future<ConstTileType> read(const LocalTileIndex& index) noexcept;

  /// Returns a read-only shared_future of the Tile with global index @p index.
  ///
  /// TODO: Sync details.
  /// @throws std::invalid_argument if the global tile is not stored in the current process.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(globalNrTiles()) == true.
  hpx::shared_future<ConstTileType> read(const GlobalTileIndex& index) noexcept;

protected:
  /// Returns the position in the vector of the index Tile.
  ///
  /// @pre index.isValid() == true.
  /// @pre index.isIn(localNrTiles()) == true.
  std::size_t tileLinearIndex(const LocalTileIndex& index) const noexcept {
    assert(index.isValid() && index.isIn(localNrTiles()));
    using util::size_t::sum;
    using util::size_t::mul;
    return sum(index.row(), mul(localNrTiles().rows(), index.col()));
  }

private:
  Matrix(matrix::Distribution&& distribution, std::vector<hpx::future<TileType>>&& tile_futures,
         std::vector<hpx::shared_future<ConstTileType>>&& tile_shared_futures);

  void setUpTiles(const memory::MemoryView<ElementType, device>& mem,
                  const matrix::LayoutInfo& layout) noexcept;

  std::size_t futureVectorSize(const matrix::LayoutInfo& layout) const noexcept;

  std::vector<hpx::future<TileType>> tile_futures_;
  std::vector<hpx::shared_future<ConstTileType>> tile_shared_futures_;
};

#include "dlaf/matrix_const.tpp"

// Note: the templates of the following helper functions are inverted w.r.t. the Matrix templates
// to allow the user to only specify the device and let the compiler deduce the type T.

// Local versions

template <Device device, class T>
Matrix<T, device> createMatrixFromColMajor(const LocalElementSize& size,
                                           const TileElementSize& block_size, SizeType ld, T* ptr) {
  return Matrix<T, device>(matrix::colMajorLayout(size, block_size, ld), ptr);
}

template <Device device, class T>
Matrix<T, device> createMatrixFromTile(const LocalElementSize& size, const TileElementSize& block_size,
                                       T* ptr) {
  return Matrix<T, device>(matrix::tileLayout(size, block_size), ptr);
}

template <Device device, class T>
Matrix<T, device> createMatrixFromTile(const LocalElementSize& size, const TileElementSize& block_size,
                                       SizeType ld_tile, SizeType tiles_per_col, T* ptr) {
  return Matrix<T, device>(matrix::tileLayout(size, block_size, ld_tile, tiles_per_col), ptr);
}

// Distributed versions

template <Device device, class T>
Matrix<T, device> createMatrixFromColMajor(const GlobalElementSize& size,
                                           const TileElementSize& block_size, SizeType ld,
                                           const comm::CommunicatorGrid& comm,
                                           const comm::Index2D& source_rank_index, T* ptr) {
  matrix::Distribution distribution(size, block_size, comm.size(), comm.rank(), source_rank_index);
  auto layout = matrix::colMajorLayout(distribution.localSize(), block_size, ld);

  return Matrix<T, device>(std::move(distribution), layout, ptr);
}

template <Device device, class T>
Matrix<T, device> createMatrixFromColMajor(const GlobalElementSize& size,
                                           const TileElementSize& block_size, SizeType ld,
                                           const comm::CommunicatorGrid& comm, T* ptr) {
  return createMatrixFromColMajor<device>(size, block_size, ld, comm, {0, 0}, ptr);
}

template <Device device, class T>
Matrix<T, device> createMatrixFromTile(const GlobalElementSize& size, const TileElementSize& block_size,
                                       const comm::CommunicatorGrid& comm,
                                       const comm::Index2D& source_rank_index, T* ptr) {
  matrix::Distribution distribution(size, block_size, comm.size(), comm.rank(), source_rank_index);
  auto layout = matrix::tileLayout(distribution.localSize(), block_size);

  return Matrix<T, device>(std::move(distribution), layout, ptr);
}

template <Device device, class T>
Matrix<T, device> createMatrixFromTile(const GlobalElementSize& size, const TileElementSize& block_size,
                                       const comm::CommunicatorGrid& comm, T* ptr) {
  return createMatrixFromTile<device>(size, block_size, comm, {0, 0}, ptr);
}

template <Device device, class T>
Matrix<T, device> createMatrixFromTile(const GlobalElementSize& size, const TileElementSize& block_size,
                                       SizeType ld_tile, SizeType tiles_per_col,
                                       const comm::CommunicatorGrid& comm,
                                       const comm::Index2D& source_rank_index, T* ptr) {
  matrix::Distribution distribution(size, block_size, comm.size(), comm.rank(), source_rank_index);
  auto layout = matrix::tileLayout(distribution.localSize(), block_size, ld_tile, tiles_per_col);

  return Matrix<T, device>(std::move(distribution), layout, ptr);
}

template <Device device, class T>
Matrix<T, device> createMatrixFromTile(const GlobalElementSize& size, const TileElementSize& block_size,
                                       SizeType ld_tile, SizeType tiles_per_col,
                                       const comm::CommunicatorGrid& comm, T* ptr) {
  return createMatrixFromTile<device>(size, block_size, ld_tile, tiles_per_col, comm, {0, 0}, ptr);
}

}
