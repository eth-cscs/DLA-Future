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
#include "blas.hh"

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/tile.h"
#include "dlaf/types.h"

namespace dlaf {
namespace matrix {

template <class T, Device device>
class MatrixView;

template <class T, Device device>
class MatrixView<const T, device> : public matrix::internal::MatrixBase {
public:
  using ElementType = T;
  using TileType = Tile<ElementType, device>;
  using ConstTileType = Tile<const ElementType, device>;

  template <template <class, Device> class MatrixType, class T2,
            std::enable_if_t<std::is_same<T, std::remove_const_t<T2>>::value, int> = 0>
  MatrixView(blas::Uplo uplo, MatrixType<T2, device>& matrix);

  MatrixView(const MatrixView& rhs) = delete;
  MatrixView(MatrixView&& rhs) = default;

  MatrixView& operator=(const MatrixView& rhs) = delete;
  MatrixView& operator=(MatrixView&& rhs) = default;
  // MatrixView& operator=(MatrixView<T, device>&& rhs);

  /// Returns a read-only shared_future of the Tile with local index @p index.
  ///
  /// TODO: Sync details.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(distribution().localNrTiles()) == true.
  hpx::shared_future<ConstTileType> read(const LocalTileIndex& index) noexcept;

  /// Returns a read-only shared_future of the Tile with global index @p index.
  ///
  /// TODO: Sync details.
  /// When the assertion is enabled, terminates the program with an error
  /// message if @p !size.isValid(),  @throw std::invalid_argument if the global
  /// tile is not stored in the current process. This assertion is enabled when
  /// **DLAF_ASSERT_ENABLE** is ON.
  ///
  /// @pre index.isValid() == true.
  /// @pre index.isIn(globalNrTiles()) == true.
  hpx::shared_future<ConstTileType> read(const GlobalTileIndex& index) {
    return read(distribution().localTileIndex(index));
  }

  /// Notify that all the operation on the @p index tile has been performed.
  ///
  /// The tile of the original matrix gets ready.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(distribution().localNrTiles()) == true.
  /// @post any call to read() with index or the equivalent GlobalTileIndex return an invalid future.
  void done(const LocalTileIndex& index) noexcept;

  /// Notify that all the operation on the @p index tile has been performed.
  ///
  /// The tile of the original matrix gets ready.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(globalNrTiles()) == true.
  /// @post any call to read() with index or the equivalent LocalTileIndex return an invalid future.
  void done(const GlobalTileIndex& index) noexcept {
    done(distribution().localTileIndex(index));
  }

private:
  template <template <class, Device> class MatrixType, class T2,
            std::enable_if_t<std::is_same<T, std::remove_const_t<T2>>::value, int> = 0>
  void setUpTiles(MatrixType<T2, device>& matrix) noexcept;

  std::vector<hpx::shared_future<ConstTileType>> tile_shared_futures_;
};

template <template <class, Device> class MatrixType, class T, Device device>
MatrixView<std::add_const_t<T>, device> getConstView(blas::Uplo uplo, MatrixType<T, device>& matrix) {
  return MatrixView<std::add_const_t<T>, device>(uplo, matrix);
}

template <template <class, Device> class MatrixType, class T, Device device>
MatrixView<std::add_const_t<T>, device> getConstView(MatrixType<T, device>& matrix) {
  return getConstView(blas::Uplo::General, matrix);
}

#include "dlaf/matrix/matrix_view_const.tpp"

}
}
