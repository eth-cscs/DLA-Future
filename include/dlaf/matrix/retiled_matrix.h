//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <vector>

#include <pika/future.hpp>

#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/internal/tile_future_manager.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/keep_future.h"
#include "dlaf/types.h"

#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"

namespace dlaf::matrix {

/// A @c RetiledMatrix object represents a collection of tiles which contain all the elements of a matrix.
/// It is constructed on top of a Matrix and allows to divide the distribution block in multiple tiles.
///
/// The tiles are distributed according to a distribution (see @c Matrix::distribution()),
/// therefore some tiles are stored locally on this rank,
/// while the others are available on other ranks.
/// More details are available in misc/matrix_distribution.md.
/// Details about the Tile synchronization mechanism can be found in misc/synchronization.md.

template <class T, Device D>
class RetiledMatrix : public internal::MatrixBase {
public:
  static constexpr Device device = D;

  using ElementType = T;
  using TileType = Tile<ElementType, D>;
  using ConstTileType = Tile<const ElementType, D>;
  using TileDataType = internal::TileData<const ElementType, D>;

  ///
  /// @pre mat.blockSize() is divisible by tiles_per_block.
  RetiledMatrix(Matrix<T, D>& mat, const LocalTileSize& tiles_per_block)
      : MatrixBase(mat.distribution(), tiles_per_block),
        tile_managers_(to_sizet(distribution().localNrTiles().linear_size())) {
    using common::internal::vector;
    const auto tile_size = distribution().baseTileSize();
    vector<SubTileSpec> specs;
    vector<LocalTileIndex> indices;
    specs.reserve(tiles_per_block.linear_size());
    indices.reserve(tiles_per_block.linear_size());

    for (const auto& orig_tile_index : common::iterate_range2d(mat.distribution().localNrTiles())) {
      const auto original_tile_size = mat.tileSize(mat.distribution().globalTileIndex(orig_tile_index));

      for (SizeType j = 0; j < original_tile_size.cols(); j += tile_size.cols())
        for (SizeType i = 0; i < original_tile_size.rows(); i += tile_size.rows()) {
          indices.emplace_back(
              LocalTileIndex{orig_tile_index.row() * tiles_per_block.rows() + i / tile_size.rows(),
                             orig_tile_index.col() * tiles_per_block.cols() + j / tile_size.cols()});
          specs.emplace_back(
              SubTileSpec{{i, j}, tileSize(distribution().globalTileIndex(indices.back()))});
        }

      auto sub_tiles = splitTileDisjoint(mat(orig_tile_index), specs);

      DLAF_ASSERT_HEAVY(specs.size() == indices.size(), specs.size(), indices.size());
      for (SizeType j = 0; j < specs.size(); ++j) {
        const auto i = tileLinearIndex(indices[j]);
        tile_managers_[i] = internal::SplittedTileFutureManager<T, D>(std::move(sub_tiles[to_sizet(j)]));
      }

      specs.clear();
      indices.clear();
    }
  }

  RetiledMatrix(const RetiledMatrix& rhs) = delete;
  RetiledMatrix(RetiledMatrix&& rhs) = default;

  RetiledMatrix& operator=(const RetiledMatrix& rhs) = delete;
  RetiledMatrix& operator=(RetiledMatrix&& rhs) = default;

  /// Returns a read-only shared_future of the Tile with local index @p index.
  ///
  /// See misc/synchronization.md for the synchronization details.
  /// @pre index.isIn(distribution().localNrTiles()).
  pika::shared_future<ConstTileType> read(const LocalTileIndex& index) noexcept {
    const auto i = tileLinearIndex(index);
    return tile_managers_[i].getReadTileSharedFuture();
  }

  /// Returns a read-only shared_future of the Tile with global index @p index.
  ///
  /// See misc/synchronization.md for the synchronization details.
  /// @pre the global tile is stored in the current process,
  /// @pre index.isIn(globalNrTiles()).
  pika::shared_future<ConstTileType> read(const GlobalTileIndex& index) noexcept {
    return read(distribution().localTileIndex(index));
  }

  auto read_sender(const LocalTileIndex& index) noexcept {
    // We want to explicitly deal with the shared_future, not the const& to the
    // value.
    return dlaf::internal::keepFuture(read(index));
  }

  auto read_sender(const GlobalTileIndex& index) noexcept {
    return read_sender(distribution().localTileIndex(index));
  }

  /// Synchronization barrier for all local tiles in the matrix
  ///
  /// This blocking call does not return until all operations, i.e. both RO and RW,
  /// involving any of the locally available tiles are completed.
  void waitLocalTiles() noexcept {
    auto readwrite_f = [this](const LocalTileIndex& index) {
      const auto i = tileLinearIndex(index);
      return tile_managers_[i].getRWTileFuture();
    };

    const auto range_local = common::iterate_range2d(distribution().localNrTiles());
    pika::wait_all(internal::selectGeneric(readwrite_f, range_local));
  }

  /// Returns a future of the Tile with local index @p index.
  ///
  /// See misc/synchronization.md for the synchronization details.
  /// @pre index.isIn(distribution().localNrTiles()).
  pika::future<TileType> operator()(const LocalTileIndex& index) noexcept {
    const auto i = tileLinearIndex(index);
    return tile_managers_[i].getRWTileFuture();
  }

  /// Returns a future of the Tile with global index @p index.
  ///
  /// See misc/synchronization.md for the synchronization details.
  /// @pre the global tile is stored in the current process,
  /// @pre index.isIn(globalNrTiles()).
  pika::future<TileType> operator()(const GlobalTileIndex& index) noexcept {
    return operator()(distribution().localTileIndex(index));
  }

  auto readwrite_sender(const LocalTileIndex& index) noexcept {
    // Note: do not use `keep_future`, otherwise dlaf::transform will not handle the lifetime correctly
    return operator()(index);
  }

  auto readwrite_sender(const GlobalTileIndex& index) noexcept {
    return readwrite_sender(distribution().localTileIndex(index));
  }

  void done(const LocalTileIndex& index) noexcept {
    const auto i = tileLinearIndex(index);
    tile_managers_[i].clear();
  }

  void done(const GlobalTileIndex& index) noexcept {
    done(distribution().localTileIndex(index));
  }

protected:
  std::vector<internal::SplittedTileFutureManager<T, D>> tile_managers_;
};

/// Re-tiling with const tiles not yet supported.
template <class T, Device D>
class RetiledMatrix<const T, D>;

/// ---- ETI

#define DLAF_RETILED_MATRIX_ETI(KWORD, DATATYPE, DEVICE) \
  KWORD template class RetiledMatrix<DATATYPE, DEVICE>;

DLAF_RETILED_MATRIX_ETI(extern, float, Device::CPU)
DLAF_RETILED_MATRIX_ETI(extern, double, Device::CPU)
DLAF_RETILED_MATRIX_ETI(extern, std::complex<float>, Device::CPU)
DLAF_RETILED_MATRIX_ETI(extern, std::complex<double>, Device::CPU)

#if defined(DLAF_WITH_GPU)
DLAF_RETILED_MATRIX_ETI(extern, float, Device::GPU)
DLAF_RETILED_MATRIX_ETI(extern, double, Device::GPU)
DLAF_RETILED_MATRIX_ETI(extern, std::complex<float>, Device::GPU)
DLAF_RETILED_MATRIX_ETI(extern, std::complex<double>, Device::GPU)
#endif
}
