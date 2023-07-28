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

namespace dlaf::matrix::internal {
/// Contains information to create a sub-matrix.
using SubMatrixSpec = SubDistributionSpec;

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

  /// Create a sub-matrix of @p mat specified by @p spec.
  ///
  /// @param[in] mat is the input matrix,
  /// @param[in] spec contains the origin and size of the new matrix relative to the input matrix,
  /// @pre spec.origin.isValid(),
  /// @pre spec.size.isValid(),
  /// @pre spec.origin + spec.size <= mat.size().
  MatrixRef(Matrix<const T, D>& mat, const SubMatrixSpec& spec)
      : internal::MatrixBase(Distribution(mat.distribution(), spec)), mat_const_(mat),
        origin_(spec.origin) {}

  MatrixRef() = delete;
  MatrixRef(MatrixRef&&) = delete;
  MatrixRef(const MatrixRef&) = delete;
  MatrixRef& operator=(MatrixRef&&) = delete;
  MatrixRef& operator=(const MatrixRef&) = delete;

  /// Returns a read-only sender of the Tile with local index @p index.
  ///
  /// @pre index.isIn(distribution().localNrTiles()).
  ReadOnlySenderType read(const LocalTileIndex& index) noexcept {
    // Note: this forwards to the overload with GlobalTileIndex which will
    // handle taking a subtile if needed
    return read(distribution().globalTileIndex(index));
  }

  /// Returns a read-only sender of the Tile with global index @p index.
  ///
  /// @pre the global tile is stored in the current process,
  /// @pre index.isIn(globalNrTiles()).
  ReadOnlySenderType read(const GlobalTileIndex& index) {
    DLAF_ASSERT(index.isIn(distribution().nrTiles()), index, distribution().nrTiles());

    const auto parent_index(
        mat_const_.distribution().globalTileIndexFromSubDistribution(origin_, distribution(), index));
    auto tile_sender = mat_const_.read(parent_index);

    const auto parent_dist = mat_const_.distribution();
    const auto parent_tile_size = parent_dist.tileSize(parent_index);
    const auto tile_size = tileSize(index);

    // If the corresponding tile in the parent distribution is exactly the same
    // size as the tile in the sub-distribution, we don't need to take a subtile
    // and can return the tile sender directly. This avoids unnecessary wrapping.
    if (parent_tile_size == tile_size) {
      return tile_sender;
    }

    // Otherwise we have to extract a subtile from the tile in the parent
    // distribution.
    const auto ij_tile =
        parent_dist.tileElementOffsetFromSubDistribution(origin_, distribution(), index);
    return splitTile(std::move(tile_sender), SubTileSpec{ij_tile, tile_size});
  }

private:
  Matrix<const T, D>& mat_const_;

protected:
  GlobalElementIndex origin_;
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

  /// Create a sub-matrix of @p mat specified by @p spec.
  ///
  /// @param[in] mat is the input matrix,
  /// @param[in] spec contains the origin and size of the new matrix relative to the input matrix,
  /// @pre spec.origin.isValid(),
  /// @pre spec.size.isValid(),
  /// @pre spec.origin + spec.size <= mat.size().
  MatrixRef(Matrix<T, D>& mat, const SubMatrixSpec& spec)
      : MatrixRef<const T, D>(mat, spec), mat_(mat) {}

  MatrixRef() = delete;
  MatrixRef(MatrixRef&&) = delete;
  MatrixRef(const MatrixRef&) = delete;
  MatrixRef& operator=(MatrixRef&&) = delete;
  MatrixRef& operator=(const MatrixRef&) = delete;

  /// Returns a sender of the Tile with local index @p index.
  ///
  /// @pre index.isIn(distribution().localNrTiles()).
  ReadWriteSenderType readwrite(const LocalTileIndex& index) noexcept {
    // Note: this forwards to the overload with GlobalTileIndex which will
    // handle taking a subtile if needed
    return readwrite(this->distribution().globalTileIndex(index));
  }

  /// Returns a sender of the Tile with global index @p index.
  ///
  /// @pre the global tile is stored in the current process,
  /// @pre index.isIn(globalNrTiles()).
  ReadWriteSenderType readwrite(const GlobalTileIndex& index) {
    DLAF_ASSERT(index.isIn(this->distribution().nrTiles()), index, this->distribution().nrTiles());

    const auto parent_index(
        mat_.distribution().globalTileIndexFromSubDistribution(origin_, this->distribution(), index));
    auto tile_sender = mat_.readwrite(parent_index);

    const auto parent_dist = mat_.distribution();
    const auto parent_tile_size = parent_dist.tileSize(parent_index);
    const auto tile_size = this->tileSize(index);

    // If the corresponding tile in the parent distribution is exactly the same
    // size as the tile in the sub-distribution, we don't need to take a subtile
    // and can return the tile sender directly. This avoids unnecessary wrapping.
    if (parent_tile_size == tile_size) {
      return tile_sender;
    }

    // Otherwise we have to extract a subtile from the tile in the parent
    // distribution.
    const auto ij_tile =
        parent_dist.tileElementOffsetFromSubDistribution(origin_, this->distribution(), index);
    return splitTile(std::move(tile_sender), SubTileSpec{ij_tile, tile_size});
  }

private:
  Matrix<T, D>& mat_;
  using MatrixRef<const T, D>::origin_;
};

// ETI

#define DLAF_MATRIX_REF_ETI(KWORD, DATATYPE, DEVICE) \
  KWORD template class MatrixRef<DATATYPE, DEVICE>;  \
  KWORD template class MatrixRef<const DATATYPE, DEVICE>;

DLAF_MATRIX_REF_ETI(extern, float, Device::CPU)
DLAF_MATRIX_REF_ETI(extern, double, Device::CPU)
DLAF_MATRIX_REF_ETI(extern, std::complex<float>, Device::CPU)
DLAF_MATRIX_REF_ETI(extern, std::complex<double>, Device::CPU)

#if defined(DLAF_WITH_GPU)
DLAF_MATRIX_REF_ETI(extern, float, Device::GPU)
DLAF_MATRIX_REF_ETI(extern, double, Device::GPU)
DLAF_MATRIX_REF_ETI(extern, std::complex<float>, Device::GPU)
DLAF_MATRIX_REF_ETI(extern, std::complex<double>, Device::GPU)
#endif
}
