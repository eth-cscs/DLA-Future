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

#include <dlaf/common/callable_object.h>
#include <dlaf/eigensolver/tridiag_solver/coltype.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/copy_tile.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/types.h>

#ifdef DLAF_WITH_GPU
#include <whip.hpp>

#include <dlaf/eigensolver/tridiag_solver/gpu/kernels.h>
#include <dlaf/gpu/lapack/api.h>
#endif

namespace dlaf::eigensolver::internal {

template <class T>
void castToComplex(const matrix::Tile<const T, Device::CPU>& in,
                   const matrix::Tile<std::complex<T>, Device::CPU>& out) {
  DLAF_ASSERT_HEAVY(in.size() == out.size(), in.size(), out.size());
  for (auto el_idx : iterate_range2d(out.size())) {
    out(el_idx) = std::complex<T>(in(el_idx), 0);
  }
}

#ifdef DLAF_WITH_GPU
template <class T>
void castToComplex(const matrix::Tile<const T, Device::GPU>& in,
                   const matrix::Tile<std::complex<T>, Device::GPU>& out, whip::stream_t stream) {
  DLAF_ASSERT_HEAVY(in.size() == out.size(), in.size(), out.size());
  const SizeType m = in.size().rows();
  const SizeType n = in.size().cols();
  gpu::castToComplex(m, n, in.ptr(), in.ld(), out.ptr(), out.ld(), stream);
}
#endif

DLAF_MAKE_CALLABLE_OBJECT(castToComplex);

// Cuppen's decomposition
//
// Substracts the offdiagonal element at the split from the top and bottom diagonal elements and
// returns the offdiagonal element. The split is between the last row of the top tile and the first
// row of the bottom tile.
//
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

DLAF_MAKE_CALLABLE_OBJECT(cuppensDecomp);

template <class T>
void copyDiagonalFromCompactTridiagonal(const matrix::Tile<const T, Device::CPU>& tridiag_tile,
                                        const matrix::Tile<T, Device::CPU>& diag_tile) {
  const SizeType m = tridiag_tile.size().rows();
  DLAF_ASSERT_HEAVY(m == diag_tile.size().rows(), m, diag_tile.size().rows());
  for (SizeType i = 0; i < m; ++i) {
    diag_tile(TileElementIndex(i, 0)) = tridiag_tile(TileElementIndex(i, 0));
  }
}

#ifdef DLAF_WITH_GPU
template <class T>
void copyDiagonalFromCompactTridiagonal(const matrix::Tile<const T, Device::CPU>& tridiag_tile,
                                        const matrix::Tile<T, Device::GPU>& diag_tile,
                                        whip::stream_t stream) {
  const SizeType m = tridiag_tile.size().rows();
  DLAF_ASSERT_HEAVY(m == diag_tile.size().rows(), m, diag_tile.size().rows());

  whip::memcpy_async(diag_tile.ptr(), tridiag_tile.ptr(), sizeof(T) * to_sizet(m), whip::memcpy_default,
                     stream);
}
#endif

DLAF_MAKE_CALLABLE_OBJECT(copyDiagonalFromCompactTridiagonal);

template <class T>
void assembleRank1UpdateVectorTile(bool is_top_tile, T rho,
                                   const matrix::Tile<const T, Device::CPU>& evecs_tile,
                                   const matrix::Tile<T, Device::CPU>& rank1_tile) {
  const SizeType m = evecs_tile.size().rows();
  const SizeType n = evecs_tile.size().cols();
  DLAF_ASSERT_HEAVY(n == rank1_tile.size().rows(), n, rank1_tile.size().rows());

  // Copy the bottom row of the top tile or the top row of the bottom tile
  SizeType row = (is_top_tile) ? m - 1 : 0;

  // Negate the last row of the top eigenvector subproblem matrix (Q1) if rho < 0
  //
  // lapack 3.10.0, dlaed2.f, line 280 and 281
  int sign = (is_top_tile && rho < 0) ? -1 : 1;

  for (SizeType i = 0; i < n; ++i) {
    rank1_tile(TileElementIndex(i, 0)) = sign * evecs_tile(TileElementIndex(row, i)) / T(std::sqrt(2));
  }
}

#ifdef DLAF_WITH_GPU

template <class T>
void assembleRank1UpdateVectorTile(bool is_top_tile, T rho,
                                   const matrix::Tile<const T, Device::GPU>& evecs_tile,
                                   const matrix::Tile<T, Device::GPU>& rank1_tile,
                                   whip::stream_t stream) {
  const SizeType m = evecs_tile.size().rows();
  const SizeType n = evecs_tile.size().cols();
  DLAF_ASSERT_HEAVY(n == rank1_tile.size().rows(), n, rank1_tile.size().rows());

  gpu::assembleRank1UpdateVectorTile(is_top_tile, rho, m, n, evecs_tile.ptr(), evecs_tile.ld(),
                                     rank1_tile.ptr(), stream);
}
#endif

DLAF_MAKE_CALLABLE_OBJECT(assembleRank1UpdateVectorTile);

#ifdef DLAF_WITH_GPU
using gpu::givensRotationOnDevice;
#endif
}
