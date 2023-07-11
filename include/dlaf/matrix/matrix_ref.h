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

/// @file

#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_base.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/types.h>

namespace dlaf::matrix {
/// A @c MatrixRef represents a sub-matrix of a @c Matrix.
///
/// The class has reference semantics, meaning accesses to a @c MatrixRef and
/// it's corresponding @c Matrix are interleaved if calls to read/readwrite are
/// interleaved. Access to a @c MatrixRef and its corresponding @c Matrix is not
/// thread-safe. A @c MatrixRef must outlive its corresponding @c Matrix.
template <class T, Device D>
class MatrixRef;

template <class T, Device D>
class MatrixRef<const T, D> : public internal::MatrixBase {
public:
  static constexpr Device device = D;

  using ElementType = T;
  using TileType = Tile<ElementType, D>;
  using ConstTileType = Tile<const ElementType, D>;
  using TileDataType = internal::TileData<ElementType, D>;
  using ReadOnlySenderType = ReadOnlyTileSender<T, D>;

  /// Create a sub-matrix of @p mat with an @p offset and @p size.
  ///
  /// @param[in] mat is the input matrix,
  /// @param[in] offset is the offset of the new matrix relative to the input matrix,
  /// @param[in] size is the size of the new matrix relative to the offset,
  /// @pre origin.isValid()
  /// @pre size.isValid()
  /// @pre origin + size <= mat.size()
  MatrixRef(Matrix<const T, D>& mat, const GlobalElementIndex& offset, const GlobalElementSize& size)
      : internal::MatrixBase(Distribution(mat.distribution(), offset, size)), mat_const_(mat),
        offset_(offset) {}

  // TODO: default, copy, move construction?
  // - default: no, don't want empty MatrixRef
  // - copy: implementable, still refer to the original matrix
  // - move: implement as copy, i.e. still refer to original matrix?
  MatrixRef() = delete;

  /// Returns a read-only sender of the Tile with local index @p index.
  ///
  /// @pre index.isIn(distribution().localNrTiles()).
  ReadOnlySenderType read(const LocalTileIndex& index) noexcept {
    DLAF_ASSERT(index.isIn(distribution().localNrTiles()), index, distribution().localNrTiles());

    // Note: the overload with GlobalTileIndex handles taking a subtile if needed
    const GlobalTileIndex global_index(distribution().globalTileFromLocalTile<Coord::Row>(index.row()),
                                       distribution().globalTileFromLocalTile<Coord::Col>(index.col()));
    return read(global_index);
  }

  /// Returns a read-only sender of the Tile with global index @p index.
  ///
  /// @pre the global tile is stored in the current process,
  /// @pre index.isIn(globalNrTiles()).
  ReadOnlySenderType read(const GlobalTileIndex& index) {
    DLAF_ASSERT(index.isIn(distribution().nrTiles()), index, distribution().nrTiles());

    const GlobalTileIndex tile_offset = mat_const_.distribution().globalTileIndex(offset_);
    const GlobalTileIndex parent_index(tile_offset.row() + index.row(), tile_offset.col() + index.col());
    auto tile_sender = mat_const_.read(parent_index);

    const auto parent_dist = mat_const_.distribution();
    const auto parent_tile_size = parent_dist.tileSize(parent_index);
    const auto tile_size = tileSize(index);

    // If the corresponding tile in the parent distribution is exactly the same
    // size as the tile in the sub-distribution, we don't need to take a subtile
    // and can return the tile sender directly.
    if (parent_tile_size == tile_size) {
      return tile_sender;
    }

    // Otherwise we have to extract a subtile from the tile in the parent
    // distribution.
    const TileElementIndex ij_tile{
        index.row() == 0 ? parent_dist.template tileElementFromGlobalElement<Coord::Row>(offset_.row())
                         : 0,
        index.col() == 0 ? parent_dist.template tileElementFromGlobalElement<Coord::Col>(offset_.col())
                         : 0,
    };

    return splitTile(std::move(tile_sender), SubTileSpec{ij_tile, tile_size});
  }

private:
  Matrix<const T, D>& mat_const_;

protected:
  GlobalElementIndex offset_;
};

template <class T, Device D>
class MatrixRef : public MatrixRef<const T, D> {
public:
  static constexpr Device device = D;

  using ElementType = T;
  using TileType = Tile<ElementType, D>;
  using ConstTileType = Tile<const ElementType, D>;
  using TileDataType = internal::TileData<ElementType, D>;
  using ReadWriteSenderType = ReadWriteTileSender<T, D>;

  /// Create a sub-matrix of @p mat with an @p offset and @p size.
  ///
  /// @param[in] mat is the input matrix,
  /// @param[in] offset is the offset of the new matrix relative to the input matrix,
  /// @param[in] size is the size of the new matrix relative to the offset,
  /// @pre origin.isValid()
  /// @pre size.isValid()
  /// @pre origin + size <= mat.size()
  MatrixRef(Matrix<T, D>& mat, const GlobalElementIndex& offset, const GlobalElementSize& size)
      : MatrixRef<const T, D>(mat, offset, size), mat_(mat) {}

  // TODO: default, copy, move construction?
  MatrixRef() = delete;

  /// Returns a sender of the Tile with local index @p index.
  ///
  /// @pre index.isIn(distribution().localNrTiles()).
  ReadWriteSenderType readwrite(const LocalTileIndex& index) noexcept {
    DLAF_ASSERT(index.isIn(this->distribution().localNrTiles()), index,
                this->distribution().localNrTiles());
    const GlobalTileIndex parent_index(
        this->distribution().template globalTileFromLocalTile<Coord::Row>(index.row()),
        this->distribution().template globalTileFromLocalTile<Coord::Col>(index.col()));

    // Note: the overload with GlobalTileIndex handles taking a subtile if needed
    return readwrite(parent_index);
  }

  /// Returns a sender of the Tile with global index @p index.
  ///
  /// @pre the global tile is stored in the current process,
  /// @pre index.isIn(globalNrTiles()).
  ReadWriteSenderType readwrite(const GlobalTileIndex& index) {
    DLAF_ASSERT(index.isIn(this->distribution().nrTiles()), index, this->distribution().nrTiles());
    // TODO: add helpers for distribution vs. sub-distribution arithmetic

    const GlobalTileIndex tile_offset = mat_.distribution().globalTileIndex(offset_);
    const GlobalTileIndex parent_index(tile_offset.row() + index.row(), tile_offset.col() + index.col());
    auto tile_sender = mat_.readwrite(parent_index);

    const auto parent_dist = mat_.distribution();
    const auto parent_tile_size = parent_dist.tileSize(parent_index);
    const auto tile_size = this->tileSize(index);

    // If the corresponding tile in the parent distribution is exactly the same
    // size as the tile in the sub-distribution, we don't need to take a subtile
    // and can return the tile sender directly.
    if (parent_tile_size == tile_size) {
      return tile_sender;
    }

    // Otherwise we have to extract a subtile from the tile in the parent
    // distribution.
    const TileElementIndex ij_tile{
        index.row() == 0 ? parent_dist.template tileElementFromGlobalElement<Coord::Row>(offset_.row())
                         : 0,
        index.col() == 0 ? parent_dist.template tileElementFromGlobalElement<Coord::Col>(offset_.col())
                         : 0,
    };

    return splitTile(std::move(tile_sender), SubTileSpec{ij_tile, tile_size});
  }

private:
  Matrix<T, D>& mat_;
  using MatrixRef<const T, D>::offset_;
};
}
