//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/eigensolver/tridiag_solver/kernels.h"
#include "dlaf/common/single_threaded_blas.h"
#include "dlaf/lapack/tile.h"

namespace dlaf::eigensolver::internal {

template <class T>
void castToComplex(const matrix::Tile<const T, Device::CPU>& in,
                   const matrix::Tile<std::complex<T>, Device::CPU>& out) {
  for (auto el_idx : iterate_range2d(out.size())) {
    out(el_idx) = std::complex<T>(in(el_idx), 0);
  }
}

DLAF_CPU_CAST_TO_COMPLEX_ETI(, float);
DLAF_CPU_CAST_TO_COMPLEX_ETI(, double);

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

template <class T>
void divideEvecsByDiagonal(const SizeType& k_row, const SizeType& k_col, const SizeType& i_subm_el,
                           const SizeType& j_subm_el,
                           const matrix::Tile<const T, Device::CPU>& diag_rows,
                           const matrix::Tile<const T, Device::CPU>& diag_cols,
                           const matrix::Tile<const T, Device::CPU>& evecs_tile,
                           const matrix::Tile<T, Device::CPU>& ws_tile) {
  if (k_col <= 2 || i_subm_el >= k_row || j_subm_el >= k_col)
    return;

  SizeType nrows = std::min(k_row - i_subm_el, evecs_tile.size().rows());
  SizeType ncols = std::min(k_col - j_subm_el, evecs_tile.size().cols());

  for (SizeType j = 0; j < ncols; ++j) {
    for (SizeType i = 0; i < nrows; ++i) {
      T di = diag_rows(TileElementIndex(i, 0));
      T dj = diag_cols(TileElementIndex(j, 0));
      T evec_el = evecs_tile(TileElementIndex(i, j));
      T& ws_el = ws_tile(TileElementIndex(i, 0));

      // Exact comparison is OK because di and dj come from the same vector
      T weight = (di == dj) ? evec_el : evec_el / (di - dj);
      ws_el = (j == 0) ? weight : weight * ws_el;
    }
  }
}

DLAF_CPU_DIVIDE_EVECS_BY_DIAGONAL_ETI(, float);
DLAF_CPU_DIVIDE_EVECS_BY_DIAGONAL_ETI(, double);

template <class T>
void multiplyFirstColumns(const SizeType& k_row, const SizeType& k_col, const SizeType& row,
                          const SizeType& col, const matrix::Tile<const T, Device::CPU>& in,
                          const matrix::Tile<T, Device::CPU>& out) {
  if (k_col <= 2 || row >= k_row || col >= k_col)
    return;

  SizeType nrows = std::min(k_row - row, in.size().rows());

  for (SizeType i = 0; i < nrows; ++i) {
    TileElementIndex idx(i, 0);
    out(idx) = out(idx) * in(idx);
  }
}

DLAF_CPU_MULTIPLY_FIRST_COLUMNS_ETI(, float);
DLAF_CPU_MULTIPLY_FIRST_COLUMNS_ETI(, double);

template <class T>
void calcEvecsFromWeightVec(const SizeType& k_row, const SizeType& k_col, const SizeType& row,
                            const SizeType& col, const matrix::Tile<const T, Device::CPU>& z_tile,
                            const matrix::Tile<const T, Device::CPU>& ws_tile,
                            const matrix::Tile<T, Device::CPU>& evecs_tile) {
  if (k_col <= 2 || row >= k_row || col >= k_col)
    return;

  SizeType nrows = std::min(k_row - row, evecs_tile.size().rows());
  SizeType ncols = std::min(k_col - col, evecs_tile.size().cols());

  for (SizeType j = 0; j < ncols; ++j) {
    for (SizeType i = 0; i < nrows; ++i) {
      T ws_el = ws_tile(TileElementIndex(i, 0));
      T z_el = z_tile(TileElementIndex(i, 0));
      T& el_evec = evecs_tile(TileElementIndex(i, j));
      // Avoid NaN generated by deflated vectors.
      // It is a temporary workaround until deflated vectors are moved at the end of the local matrix.
      if (el_evec != 0)
        el_evec = std::copysign(std::sqrt(std::abs(ws_el)), z_el) / el_evec;
    }
  }
}

DLAF_CPU_CALC_EVECS_FROM_WEIGHT_VEC_ETI(, float);
DLAF_CPU_CALC_EVECS_FROM_WEIGHT_VEC_ETI(, double);

template <class T>
void sumsqCols(const SizeType& k_row, const SizeType& k_col, const SizeType& row, const SizeType& col,
               const matrix::Tile<const T, Device::CPU>& evecs_tile,
               const matrix::Tile<T, Device::CPU>& ws_tile) {
  if (k_col <= 2 || row >= k_row || col >= k_col)
    return;

  SizeType nrows = std::min(k_row - row, evecs_tile.size().rows());
  SizeType ncols = std::min(k_col - col, evecs_tile.size().cols());

  for (SizeType j = 0; j < ncols; ++j) {
    T loc_norm = 0;
    for (SizeType i = 0; i < nrows; ++i) {
      T el = evecs_tile(TileElementIndex(i, j));
      loc_norm += el * el;
    }
    ws_tile(TileElementIndex(0, j)) = loc_norm;
  }
}

DLAF_CPU_SUMSQ_COLS_ETI(, float);
DLAF_CPU_SUMSQ_COLS_ETI(, double);

template <class T>
void addFirstRows(const SizeType& k_row, const SizeType& k_col, const SizeType& row, const SizeType& col,
                  const matrix::Tile<const T, Device::CPU>& in,
                  const matrix::Tile<T, Device::CPU>& out) {
  if (k_col <= 2 || row >= k_row || col >= k_col)
    return;

  SizeType ncols = std::min(k_col - col, in.size().cols());

  for (SizeType j = 0; j < ncols; ++j) {
    out(TileElementIndex(0, j)) += in(TileElementIndex(0, j));
  }
}

DLAF_CPU_ADD_FIRST_ROWS_ETI(, float);
DLAF_CPU_ADD_FIRST_ROWS_ETI(, double);

template <class T>
void divideColsByFirstRow(const SizeType& k_row, const SizeType& k_col, const SizeType& row,
                          const SizeType& col, const matrix::Tile<const T, Device::CPU>& in,
                          const matrix::Tile<T, Device::CPU>& out) {
  if (k_col <= 2 || row >= k_row || col >= k_col)
    return;

  SizeType nrows = std::min(k_row - row, out.size().rows());
  SizeType ncols = std::min(k_col - col, out.size().cols());

  for (SizeType j = 0; j < ncols; ++j) {
    for (SizeType i = 0; i < nrows; ++i) {
      T& evecs_el = out(TileElementIndex(i, j));
      evecs_el = evecs_el / std::sqrt(in(TileElementIndex(0, j)));
    }
  }
}

DLAF_CPU_DIVIDE_COLS_BY_FIRST_ROW_ETI(, float);
DLAF_CPU_DIVIDE_COLS_BY_FIRST_ROW_ETI(, double);

template <class T>
void setUnitDiagonal(const SizeType& k, const SizeType& tile_begin,
                     const matrix::Tile<T, Device::CPU>& tile) {
  // If all elements of the tile are after the `k` index reset the offset
  SizeType tile_offset = k - tile_begin;
  if (tile_offset < 0)
    tile_offset = 0;

  // Set all diagonal elements of the tile to 1.
  for (SizeType i = tile_offset; i < tile.size().rows(); ++i) {
    tile(TileElementIndex(i, i)) = 1;
  }
}

DLAF_CPU_SET_UNIT_DIAGONAL_ETI(, float);
DLAF_CPU_SET_UNIT_DIAGONAL_ETI(, double);

template <class T>
void copy1D(const SizeType& k_row, const SizeType& k_col, const SizeType& row, const SizeType& col,
            const Coord& in_coord, const matrix::Tile<const T, Device::CPU>& in_tile,
            const Coord& out_coord, const matrix::Tile<T, Device::CPU>& out_tile) {
  dlaf::common::internal::SingleThreadedBlasScope single;

  if (k_col <= 2 || row >= k_row || col >= k_col)
    return;

  const T* in_ptr = in_tile.ptr();
  T* out_ptr = out_tile.ptr();

  SizeType in_ld = (in_coord == Coord::Col) ? 1 : in_tile.ld();
  SizeType out_ld = (out_coord == Coord::Col) ? 1 : out_tile.ld();

  // if `in_tile` is the column buffer
  SizeType len = (out_coord == Coord::Col) ? std::min(out_tile.size().rows(), k_row - row)
                                           : std::min(out_tile.size().cols(), k_col - col);
  // if out_tile is the column buffer
  if (out_tile.size().cols() == 1) {
    len = (in_coord == Coord::Col) ? std::min(in_tile.size().rows(), k_row - row)
                                   : std::min(in_tile.size().cols(), k_col - col);
  }

  blas::copy(len, in_ptr, in_ld, out_ptr, out_ld);
}

DLAF_CPU_COPY_1D_ETI(, float);
DLAF_CPU_COPY_1D_ETI(, double);

}
