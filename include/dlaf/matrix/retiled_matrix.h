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

#include <cstddef>
#include <vector>

#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/internal/tile_pipeline.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/matrix/tile.h"
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
template <class T, Device D>
class RetiledMatrix : public internal::MatrixBase {
public:
  static constexpr Device device = D;

  using ElementType = T;
  using TileType = Tile<ElementType, D>;
  using ConstTileType = Tile<const ElementType, D>;
  using TileDataType = internal::TileData<const ElementType, D>;

  /// @pre mat.blockSize() is divisible by tiles_per_block.
  RetiledMatrix(Matrix<T, D>& mat, const LocalTileSize& tiles_per_block)
      : MatrixBase(mat.distribution(), tiles_per_block) {
    using common::internal::vector;
    namespace ex = pika::execution::experimental;

    const auto n = to_sizet(distribution().localNrTiles().linear_size());
    tile_managers_.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      tile_managers_.emplace_back(Tile<T, D>());
    }

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

      auto sub_tiles = splitTileDisjoint(mat.readwrite(orig_tile_index), specs);

      DLAF_ASSERT_HEAVY(specs.size() == indices.size(), specs.size(), indices.size());
      for (SizeType j = 0; j < specs.size(); ++j) {
        const auto i = tileLinearIndex(indices[j]);

        // Move subtile to be managed by the tile manager of RetiledMatrix. We
        // use readwrite_with_wrapper to get access to the original tile managed
        // by the underlying async_rw_mutex.
        auto s =
            ex::when_all(tile_managers_[i].readwrite_with_wrapper(), std::move(sub_tiles[to_sizet(j)])) |
            ex::then([](internal::TileAsyncRwMutexReadWriteWrapper<T, D> empty_tile_wrapper,
                        Tile<T, D> sub_tile) { empty_tile_wrapper.get() = std::move(sub_tile); });
        ex::start_detached(std::move(s));
      }

      specs.clear();
      indices.clear();
    }
  }

  RetiledMatrix(const RetiledMatrix& rhs) = delete;
  RetiledMatrix(RetiledMatrix&& rhs) = default;

  RetiledMatrix& operator=(const RetiledMatrix& rhs) = delete;
  RetiledMatrix& operator=(RetiledMatrix&& rhs) = default;

  /// Returns a read-only sender of the Tile with local index @p index.
  ///
  /// @pre index.isIn(distribution().localNrTiles()).
  ReadOnlyTileSender<T, D> read(const LocalTileIndex& index) noexcept {
    const auto i = tileLinearIndex(index);
    return tile_managers_[i].read();
  }

  /// Returns a read-only sender of the Tile with global index @p index.
  ///
  /// @pre the global tile is stored in the current process,
  /// @pre index.isIn(globalNrTiles()).
  ReadOnlyTileSender<T, D> read(const GlobalTileIndex& index) noexcept {
    return read(distribution().localTileIndex(index));
  }

  /// Synchronization barrier for all local tiles in the matrix
  ///
  /// This blocking call does not return until all operations, i.e. both RO and RW,
  /// involving any of the locally available tiles are completed.
  void waitLocalTiles() noexcept {
    auto readwrite_f = [this](const LocalTileIndex& index) {
      return this->tile_managers_[tileLinearIndex(index)].readwrite();
    };

    const auto range_local = common::iterate_range2d(distribution().localNrTiles());
    pika::this_thread::experimental::sync_wait(pika::execution::experimental::when_all_vector(
        internal::selectGeneric(readwrite_f, range_local)));
  }

  /// Returns a sender of the Tile with local index @p index.
  ///
  /// @pre index.isIn(distribution().localNrTiles()).
  ReadWriteTileSender<T, D> readwrite(const LocalTileIndex& index) noexcept {
    const auto i = tileLinearIndex(index);
    return tile_managers_[i].readwrite();
  }

  /// Returns a sender of the Tile with global index @p index.
  ///
  /// @pre the global tile is stored in the current process,
  /// @pre index.isIn(globalNrTiles()).
  ReadWriteTileSender<T, D> readwrite(const GlobalTileIndex& index) noexcept {
    return readwrite(distribution().localTileIndex(index));
  }

  void done(const LocalTileIndex& index) noexcept {
    const auto i = tileLinearIndex(index);
    tile_managers_[i].reset();
  }

  void done(const GlobalTileIndex& index) noexcept {
    done(distribution().localTileIndex(index));
  }

protected:
  std::vector<internal::TilePipeline<T, D>> tile_managers_;
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
