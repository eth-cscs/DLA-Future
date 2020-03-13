//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <vector>

#include <blas.hh>
#include <hpx/local/future.hpp>

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"

namespace dlaf {
namespace matrix {

template <class T, Device device>
class MatrixView : public internal::MatrixBase {
public:
  using ElementType = T;
  using TileType = Tile<ElementType, device>;
  using ConstTileType = Tile<const ElementType, device>;

  template <template <class, Device> class MatrixType>
  MatrixView(blas::Uplo uplo, MatrixType<T, device>& matrix, bool force_RW) : MatrixBase(matrix) {
    if (uplo != blas::Uplo::General)
      DLAF_UNIMPLEMENTED(uplo);
    setUpTiles(matrix, force_RW);
  }

  MatrixView(const MatrixView& rhs) = delete;
  MatrixView(MatrixView&& rhs) = default;

  MatrixView& operator=(const MatrixView& rhs) = delete;
  MatrixView& operator=(MatrixView&& rhs) = default;

  /// Returns a read-only shared_future of the Tile with local index @p index.
  ///
  /// TODO: Sync details.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(distribution().localNrTiles()) == true.
  hpx::shared_future<ConstTileType> read(const LocalTileIndex& index) noexcept;

  /// Returns a read-only shared_future of the Tile with global index @p index.
  ///
  /// TODO: Sync details.
  /// @throw std::invalid_argument if the global tile is not stored in the current process.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(globalNrTiles()) == true.
  hpx::shared_future<ConstTileType> read(const GlobalTileIndex& index) {
    return read(distribution().localTileIndex(index));
  }

  /// @brief Returns a future of the Tile with local index @p index.
  ///
  /// TODO: Sync details.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(distribution().localNrTiles()) == true.
  hpx::future<TileType> operator()(const LocalTileIndex& index) noexcept;

  /// @brief Returns a future of the Tile with global index @p index.
  ///
  /// TODO: Sync details.
  /// @throw std::invalid_argument if the global tile is not stored in the current process.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(globalNrTiles()) == true.
  hpx::future<TileType> operator()(const GlobalTileIndex& index) {
    return operator()(this->distribution().localTileIndex(index));
  }

  /// Notify that all the operations that modify the @p index tile has been performed.
  ///
  /// Read operations of the tile of the original matrix gets ready.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(distribution().localNrTiles()) == true.
  /// @post any call to operator() with index or the equivalent GlobalTileIndex
  ///       returns an invalid future.
  void doneWrite(const LocalTileIndex& index) noexcept;

  /// Notify that all the operations that modify the @p index tile has been performed.
  ///
  /// Read operations of the tile of the original matrix gets ready.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(globalNrTiles()) == true.
  /// @post any call to operator() with index or the equivalent LocalTileIndex
  ///       returns an invalid future.
  void doneWrite(const GlobalTileIndex& index) noexcept {
    done(distribution().localTileIndex(index));
  }

  /// Notify that all the operation on the @p index tile has been performed.
  ///
  /// The tile of the original matrix gets ready.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(distribution().localNrTiles()) == true.
  /// @post any call to read() or operator() with index or the equivalent GlobalTileIndex
  ///       returns an invalid future.
  void done(const LocalTileIndex& index) noexcept;

  /// Notify that all the operation on the @p index tile has been performed.
  ///
  /// The tile of the original matrix gets ready.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(globalNrTiles()) == true.
  /// @post any call to read() or operator() with index or the equivalent LocalTileIndex
  ///       returns an invalid future.
  void done(const GlobalTileIndex& index) noexcept {
    done(distribution().localTileIndex(index));
  }

private:
  template <template <class, Device> class MatrixType>
  void setUpTiles(MatrixType<T, device>& matrix, bool force_RW) noexcept;

  internal::ViewTileFutureManager<T, device>& tileManager(const LocalTileIndex& index) {
    std::size_t i = static_cast<std::size_t>(tileLinearIndex(index));
    return tile_managers_[i];
  }

  std::vector<internal::ViewTileFutureManager<T, device>> tile_managers_;
};

template <class T, Device device>
template <template <class, Device> class MatrixType>
void MatrixView<T, device>::setUpTiles(MatrixType<T, device>& matrix, bool force_RW) noexcept {
  const auto& nr_tiles = matrix.distribution().localNrTiles();
  tile_managers_.reserve(futureVectorSize(nr_tiles));

  for (SizeType j = 0; j < nr_tiles.cols(); ++j) {
    for (SizeType i = 0; i < nr_tiles.rows(); ++i) {
      LocalTileIndex ind(i, j);
      tile_managers_.emplace_back(matrix.tileManager(ind), force_RW);
    }
  }
}

template <template <class, Device> class MatrixType, class T, Device device,
          std::enable_if_t<!std::is_const<T>::value, int> = 0>
MatrixView<T, device> getView(blas::Uplo uplo, MatrixType<T, device>& matrix, bool force_RW = true) {
  return MatrixView<T, device>(uplo, matrix, force_RW);
}

template <template <class, Device> class MatrixType, class T, Device device,
          std::enable_if_t<!std::is_const<T>::value, int> = 0>
MatrixView<T, device> getView(MatrixType<T, device>& matrix, bool force_RW = true) {
  return getView(blas::Uplo::General, matrix, force_RW);
}

template <class T, Device device>
class MatrixView<const T, device> : public internal::MatrixBase {
public:
  using ElementType = T;
  using TileType = Tile<ElementType, device>;
  using ConstTileType = Tile<const ElementType, device>;

  template <template <class, Device> class MatrixType, class T2,
            std::enable_if_t<std::is_same<T, std::remove_const_t<T2>>::value, int> = 0>
  MatrixView(blas::Uplo uplo, MatrixType<T2, device>& matrix, bool force_R) : MatrixBase(matrix) {
    if (uplo != blas::Uplo::General)
      DLAF_UNIMPLEMENTED(uplo);
    setUpTiles(matrix, force_R);
  }

  MatrixView(const MatrixView& rhs) = delete;
  MatrixView(MatrixView&& rhs) = default;

  MatrixView& operator=(const MatrixView& rhs) = delete;
  MatrixView& operator=(MatrixView&& rhs) = default;
  // MatrixView& operator=(MatrixView<T, device>&& rhs);

  /// Returns a read-only shared_future of the Tile with local index @p index.
  ///
  /// TODO: Sync details.
  ///
  /// @pre index.isIn(distribution().localNrTiles()).
  hpx::shared_future<ConstTileType> read(const LocalTileIndex& index) noexcept;

  /// Returns a read-only shared_future of the Tile with global index @p index.
  ///
  /// TODO: Sync details.
  ///
  /// @pre index.isIn(globalNrTiles()),
  /// @pre global tile stored in current process.
  hpx::shared_future<ConstTileType> read(const GlobalTileIndex& index) noexcept {
    return read(distribution().localTileIndex(index));
  }

  /// Notify that all the operation on the @p index tile has been performed.
  ///
  /// The tile of the original matrix gets ready.
  /// @pre index.isIn(distribution().localNrTiles()),
  /// @post any call to read() with index or the equivalent GlobalTileIndex returns an invalid future.
  void done(const LocalTileIndex& index) noexcept;

  /// Notify that all the operation on the @p index tile has been performed.
  ///
  /// The tile of the original matrix gets ready.
  /// @pre index.isIn(globalNrTiles()),
  /// @post any call to read() with index or the equivalent LocalTileIndex returns an invalid future.
  void done(const GlobalTileIndex& index) noexcept {
    done(distribution().localTileIndex(index));
  }

private:
  template <template <class, Device> class MatrixType, class T2,
            std::enable_if_t<std::is_same<T, std::remove_const_t<T2>>::value, int> = 0>
  void setUpTiles(MatrixType<T2, device>& matrix, bool force_R) noexcept;

  std::vector<hpx::shared_future<ConstTileType>> tile_shared_futures_;
};

template <class T, Device device>
template <template <class, Device> class MatrixType, class T2,
          std::enable_if_t<std::is_same<T, std::remove_const_t<T2>>::value, int>>
void MatrixView<const T, device>::setUpTiles(MatrixType<T2, device>& matrix, bool force_R) noexcept {
  const auto& nr_tiles = matrix.distribution().localNrTiles();
  tile_shared_futures_.reserve(futureVectorSize(nr_tiles));

  for (SizeType j = 0; j < nr_tiles.cols(); ++j) {
    for (SizeType i = 0; i < nr_tiles.rows(); ++i) {
      LocalTileIndex ind(i, j);
      if (force_R)
        assert(matrix.read(ind).valid());

      tile_shared_futures_.emplace_back(matrix.read(ind));
    }
  }
}

template <template <class, Device> class MatrixType, class T, Device device>
MatrixView<std::add_const_t<T>, device> getConstView(blas::Uplo uplo, MatrixType<T, device>& matrix,
                                                     bool force_R = true) {
  return MatrixView<std::add_const_t<T>, device>(uplo, matrix, force_R);
}

template <template <class, Device> class MatrixType, class T, Device device>
MatrixView<std::add_const_t<T>, device> getConstView(MatrixType<T, device>& matrix,
                                                     bool force_R = true) {
  return getConstView(blas::Uplo::General, matrix, force_R);
}

}
}

#include "dlaf/matrix/matrix_view.tpp"
#include "dlaf/matrix/matrix_view_const.tpp"
