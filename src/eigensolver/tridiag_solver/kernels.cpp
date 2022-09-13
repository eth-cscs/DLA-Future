//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/eigensolver/tridiag_solver/kernels.h"
#include "dlaf/lapack/tile.h"

namespace dlaf::eigensolver::internal {

template <class T>
T cuppensDecomp(const matrix::Tile<T, Device::CPU>& top, const matrix::Tile<T, Device::CPU>& bottom) {
  TileElementIndex offdiag_idx{top.size().rows() - 1, 1};
  TileElementIndex top_idx{top.size().rows() - 1, 0};
  TileElementIndex bottom_idx{0, 0};
  const T offdiag_val = top(offdiag_idx);

  // Refence: Lapack working notes: LAWN 69, Serial Cuppen algorithm, Chapter 3
  //
  top(top_idx) -= std::abs(offdiag_val);
  bottom(bottom_idx) -= std::abs(offdiag_val);
  return offdiag_val;
}

DLAF_CPU_CUPPENS_DECOMP_ETI(, float);
DLAF_CPU_CUPPENS_DECOMP_ETI(, double);

template <class T>
void copyDiagonalFromCompactTridiagonal(const matrix::Tile<const T, Device::CPU>& tridiag_tile,
                                        const matrix::Tile<T, Device::CPU>& diag_tile) {
  for (SizeType i = 0; i < tridiag_tile.size().rows(); ++i) {
    diag_tile(TileElementIndex(i, 0)) = tridiag_tile(TileElementIndex(i, 0));
  }
}

DLAF_CPU_COPY_DIAGONAL_FROM_COMPACT_TRIDIAGONAL_ETI(, float);
DLAF_CPU_COPY_DIAGONAL_FROM_COMPACT_TRIDIAGONAL_ETI(, double);

template <class T>
void assembleRank1UpdateVectorTile(bool is_top_tile, T rho,
                                   const matrix::Tile<const T, Device::CPU>& evecs_tile,
                                   const matrix::Tile<T, Device::CPU>& rank1_tile) {
  // Copy the bottom row of the top tile or the top row of the bottom tile
  SizeType row = (is_top_tile) ? rank1_tile.size().rows() - 1 : 0;

  // Negate the last row of the top eigenvector subproblem matrix (Q1) if rho < 0
  //
  // lapack 3.10.0, dlaed2.f, line 280 and 281
  int sign = (is_top_tile && rho < 0) ? -1 : 1;

  for (SizeType i = 0; i < evecs_tile.size().cols(); ++i) {
    rank1_tile(TileElementIndex(i, 0)) = sign * evecs_tile(TileElementIndex(row, i)) / T(std::sqrt(2));
  }
}

DLAF_CPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(, float);
DLAF_CPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(, double);

template <class T>
T maxElementInColumnTile(const matrix::Tile<const T, Device::CPU>& tile) {
  return tile::internal::lange(lapack::Norm::Max, tile);
}

DLAF_CPU_MAX_ELEMENT_IN_COLUMN_TILE_ETI(, float);
DLAF_CPU_MAX_ELEMENT_IN_COLUMN_TILE_ETI(, double);

void setColTypeTile(const ColType& ct, const matrix::Tile<ColType, Device::CPU>& tile) {
  for (SizeType i = 0; i < tile.size().rows(); ++i) {
    tile(TileElementIndex(i, 0)) = ct;
  }
}

void initIndexTile(SizeType offset, const matrix::Tile<SizeType, Device::CPU>& tile) {
  for (SizeType i = 0; i < tile.size().rows(); ++i) {
    tile(TileElementIndex(i, 0)) = offset + i;
  }
}

}
