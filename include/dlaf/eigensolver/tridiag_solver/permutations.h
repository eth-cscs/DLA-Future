//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <vector>

#include "dlaf/blas/tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"

namespace dlaf::eigensolver::internal {

// Get the index of the element local to the tile that contains it as well as the index of that tile
// in the tiles arrays.
inline std::pair<TileElementIndex, std::size_t> getSubmatrixElementAndTileIndices(
    GlobalElementIndex idx_el, const matrix::Distribution& distr) {
  auto arr_idx = distr.globalTileIndex(idx_el);
  return std::make_pair(distr.tileElementIndex(idx_el),
                        arr_idx.row() + arr_idx.col() * distr.nrTiles().rows());
}

// Applies the permutaton index `perm_arr` to a portion of the columns/rows(depends on coord) [1] of an
// input submatrix [2] and saves the result into a subregion [3] of an output submatrix [4].
//
// Example column permutations with `perm_arr = [8, 2, 5]`:
//
//          2     5     8      out_begin
//     ┌────────────────────┐    ┌─────┬─────────────┐
//     │  in_offset      in │    │     │         out │
//     │ │                  │    │     └─►┌─┬─┬─┐    │
//     │◄┘ ┌─┐   ┌─┐   ┌─┐  │    │        │c│a│b│    │
//     │   │a│   │b│   │c│  │    │    ┌──►│ │ │ │    │
//     │   │ │ ┌►│ │   │ │◄─┼────┼──┐ │   └─┴─┴─┘    │
//     │   └─┘ │ └─┘   └─┘  │    │  │ │      ▲       │
//     │       │            │    │sz.rows()  │       │
//     │      sz.rows()     │    │           └─      │
//     │                    │    │          sz.cols()│
//     └────────────────────┘    └───────────────────┘
//
// Example row permutations with `perm_arr = [3, 1]`:
//
//             ┌─── in_offset
//             │                   out_begin
//     ┌───────▼────────────┐    ┌──┬────────────────┐
//     │                 in │    │  │            out │
//     │       ┌─────┐      │    │  └►┌─────┐        │
//   1 │       │  a  │      │    │    │  b  │        │
//     │       └─────┘      │    │    ├─────┤ ◄─┐    │
//     │                    │    │    │  a  │   │    │
//     │       ┌─────┐      │    │    └──▲──┘   │    │
//   3 │       │  b  │      │    │       │      │    │
//     │       └──▲──┘      │    │ sz.cols()    │    │
//     │          │         │    │          sz.rows()│
//     └──────────┼─────────┘    └───────────────────┘
//                │
//             sz.cols()
//
//
// [1]: The portion of each input column or row is defined by the interval [in_offset, in_offset +
//      sz.col()) or the interval [in_offset, in_offset + sz.row()) respectively.
// [2]: The input submatrix is defined by `begin_tiles`, `ld_tiles`, `distr` and `in_tiles`
// [3]: The subregion is defined by `begin` and `sz`
// [4]: The output submatrix is defined by `begin_tiles`, `ld_tiles`, `distr` and `out_tiles`
//
// clang-format off
template<class T, Coord coord>
void applyPermutations(
    GlobalElementIndex out_begin,
    GlobalElementSize sz,
    SizeType in_offset,
    const matrix::Distribution& distr,
    const SizeType* perm_arr,
    const std::vector<matrix::Tile<T, Device::CPU>>& in_tiles,
    std::vector<matrix::Tile<T, Device::CPU>>& out_tiles
    ) {
  // clang-format on

  // Iterate over rows if `coord == Coord::Row` otherwise iterate over columns
  SizeType dim1 = sz.cols();
  SizeType dim2 = sz.rows();
  if constexpr (coord == Coord::Row)
    std::swap(dim1, dim2);
  for (SizeType i1 = 0; i1 < dim1; ++i1) {
    for (SizeType i2 = 0; i2 < dim2; ++i2) {
      SizeType i = i2;  // row
      SizeType j = i1;  // column
      if constexpr (coord == Coord::Row)
        std::swap(i, j);

      // Get the global index of the `out` element
      GlobalElementIndex idx_el_gl_out(out_begin.row() + i, out_begin.col() + j);
      // Get the global index of the `in` element. If `coord == Coord::Row` use the permutation index on
      // the rows, otherwise use it on the columns.
      GlobalElementIndex idx_el_gl_in(in_offset + i, perm_arr[j]);
      if constexpr (coord == Coord::Row)
        idx_el_gl_in = GlobalElementIndex(perm_arr[i], in_offset + j);

      // Get the index of the element local to the tile that contains it as well as the index of that
      // tile in the tiles arrays.
      auto [idx_el_out, idx_arr_out] = getSubmatrixElementAndTileIndices(idx_el_gl_out, distr);
      auto [idx_el_in, idx_arr_in] = getSubmatrixElementAndTileIndices(idx_el_gl_in, distr);
      //  copy from `in` to `out`
      out_tiles[idx_arr_out](idx_el_out) = in_tiles[idx_arr_in](idx_el_in);
    }
  }
}

}
