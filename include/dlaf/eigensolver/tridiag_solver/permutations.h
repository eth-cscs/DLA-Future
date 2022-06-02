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

#include "pika/parallel/algorithms/for_each.hpp"

#include "dlaf/blas/tile.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"

namespace dlaf::eigensolver::internal {

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
template <class T, Coord coord>
void applyPermutations(GlobalElementIndex out_begin, GlobalElementSize sz, SizeType in_offset,
                       const matrix::Distribution& distr, const SizeType* perm_arr,
                       const std::vector<matrix::Tile<T, Device::CPU>>& in_tiles,
                       std::vector<matrix::Tile<T, Device::CPU>>& out_tiles) {
  constexpr Coord ocrd = orthogonal(coord);
  std::vector<SizeType> splits =
      util::interleaveSplits(sz.get<ocrd>(), distr.blockSize().get<ocrd>(),
                             distr.distanceToAdjacentTile<ocrd>(in_offset),
                             distr.distanceToAdjacentTile<ocrd>(out_begin.get<ocrd>()));

  for (SizeType i_perm = 0; i_perm < sz.get<coord>(); ++i_perm) {
    for (std::size_t i_split = 0; i_split < splits.size() - 1; ++i_split) {
      SizeType split = splits[i_split];

      GlobalElementIndex i_split_gl_in(split + in_offset, perm_arr[i_perm]);
      GlobalElementIndex i_split_gl_out(split + out_begin.get<ocrd>(), out_begin.get<coord>() + i_perm);
      TileElementSize region(splits[i_split + 1] - split, 1);
      if constexpr (coord == Coord::Row) {
        region.transpose();
        i_split_gl_in.transpose();
        i_split_gl_out.transpose();
      }

      TileElementIndex i_subtile_in = distr.tileElementIndex(i_split_gl_in);
      auto& tile_in = in_tiles[to_sizet(distr.globalTileLinearIndex(i_split_gl_in))];
      TileElementIndex i_subtile_out = distr.tileElementIndex(i_split_gl_out);
      auto& tile_out = out_tiles[to_sizet(distr.globalTileLinearIndex(i_split_gl_out))];

      matrix::internal::copy(region, i_subtile_in, tile_in, i_subtile_out, tile_out);
    }
  }
}

}
