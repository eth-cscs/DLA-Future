//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <vector>

#include <dlaf/blas/tile.h>
#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/types.h>

namespace dlaf::eigensolver::internal {

// Extract x0 and compute local cumulative sum of squares of the reflector column
template <Device D, class T>
std::array<T, 2> computeX0AndSquares(const bool has_head, const std::vector<matrix::Tile<T, D>>& panel,
                                     SizeType i, SizeType j, const SizeType first_tile = 0) {
  std::array<T, 2> x0_and_squares{0, 0};
  auto it_begin = std::next(panel.begin(), first_tile);
  auto it_end = panel.end();

  common::internal::SingleThreadedBlasScope single;

  if (has_head) {
    auto& tile_v0 = *it_begin++;

    const TileElementIndex idx_x0(i, j);
    x0_and_squares[0] = tile_v0(idx_x0);

    T* reflector_ptr = tile_v0.ptr({idx_x0});
    x0_and_squares[1] =
        blas::dot(tile_v0.size().rows() - idx_x0.row(), reflector_ptr, 1, reflector_ptr, 1);
  }

  for (auto it = it_begin; it != it_end; ++it) {
    const auto& tile = *it;

    T* reflector_ptr = tile.ptr({0, j});
    x0_and_squares[1] += blas::dot(tile.size().rows(), reflector_ptr, 1, reflector_ptr, 1);
  }
  return x0_and_squares;
}

template <Device D, class T>
T computeReflectorAndTau(const bool has_head, const std::vector<matrix::Tile<T, D>>& panel,
                         const SizeType i, const SizeType j, std::array<T, 2> x0_and_squares,
                         const SizeType first_tile = 0) {
  if (x0_and_squares[1] == T(0))
    return T(0);

  const T norm = std::sqrt(x0_and_squares[1]);
  const T x0 = x0_and_squares[0];
  const T y = std::signbit(std::real(x0_and_squares[0])) ? norm : -norm;
  const T tau = (y - x0) / y;

  auto it_begin = std::next(panel.begin(), first_tile);
  auto it_end = panel.end();

  common::internal::SingleThreadedBlasScope single;

  if (has_head) {
    const auto& tile_v0 = *it_begin++;

    const TileElementIndex idx_x0(i, j);
    tile_v0(idx_x0) = y;

    if (i + 1 < tile_v0.size().rows()) {
      T* v = tile_v0.ptr({i + 1, j});
      blas::scal(tile_v0.size().rows() - (i + 1), T(1) / (x0 - y), v, 1);
    }
  }

  for (auto it = it_begin; it != it_end; ++it) {
    auto& tile_v = *it;
    T* v = tile_v.ptr({0, j});
    blas::scal(tile_v.size().rows(), T(1) / (x0 - y), v, 1);
  }

  return tau;
}

}
