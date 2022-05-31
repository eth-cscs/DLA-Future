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

inline std::vector<SizeType> interleaveSplits(SizeType l, SizeType b, SizeType o1, SizeType o2) {
  DLAF_ASSERT(l > 0, l);
  DLAF_ASSERT(b > 0, b);
  DLAF_ASSERT(o1 >= 0, o1);
  DLAF_ASSERT(o2 >= 0, o2);

  // Set small and big from offsets o1 and o2 s.t small <= big
  SizeType small = o1;
  SizeType big = o2;
  if (small > big)
    std::swap(small, big);

  // Reserve enough memory for array of splits
  std::vector<SizeType> splits;
  splits.reserve(2 * to_sizet(l / b) + 2);

  splits.push_back(0);
  for (SizeType i = small, j = big; i < l || j < l; i += b, j += b) {
    if (splits.back() != i && i < l)
      splits.push_back(i);
    if (splits.back() != j && j < l)
      splits.push_back(j);
  }
  splits.push_back(l);
  return splits;
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
template <class T, Coord coord>
void applyPermutations(GlobalElementIndex out_begin, GlobalElementSize sz, SizeType in_offset,
                       const matrix::Distribution& distr, const SizeType* perm_arr,
                       const std::vector<matrix::Tile<T, Device::CPU>>& in_tiles,
                       std::vector<matrix::Tile<T, Device::CPU>>& out_tiles) {
  constexpr Coord ocrd = orthogonal(coord);
  std::vector<SizeType> splits =
      interleaveSplits(sz.get<ocrd>(), distr.blockSize().get<ocrd>(),
                       distr.distanceToAdjacentTile<ocrd>(in_offset),
                       distr.distanceToAdjacentTile<ocrd>(out_begin.get<ocrd>()));

  // for (SizeType split : splits) {
  //   std::cout << split << std::endl;
  // }

  for (SizeType i_perm = 0; i_perm < sz.get<coord>(); ++i_perm) {
    for (std::size_t i_split = 0; i_split < splits.size() - 1; ++i_split) {
      SizeType split = splits[i_split];

      GlobalElementIndex i_split_gl_in(split + in_offset, perm_arr[i_perm]);
      GlobalElementIndex i_split_gl_out(split + out_begin.get<ocrd>(), out_begin.get<coord>());
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

      // std::cout << i_subtile_in << " " << i_subtile_out << " " << region << " " << std::endl;
      std::cout << i_split_gl_in << " " << i_split_gl_out << " " << region << " " << std::endl;
      matrix::internal::copy(region, i_subtile_in, tile_in, i_subtile_out, tile_out);
    }
  }

  //  // Iterate over rows if `coord == Coord::Row` otherwise iterate over columns
  //  SizeType dim1 = sz.cols();
  //  SizeType dim2 = sz.rows();
  //  if constexpr (coord == Coord::Row)
  //    std::swap(dim1, dim2);
  //
  //  std::vector<SizeType> loop_arr(to_sizet(dim1));
  //  std::iota(std::begin(loop_arr), std::end(loop_arr), 0);
  //  pika::for_each(pika::execution::par, std::begin(loop_arr), std::end(loop_arr), [&](SizeType i1) {
  //    for (SizeType i2 = 0; i2 < dim2; ++i2) {
  //      SizeType i = i2;  // row
  //      SizeType j = i1;  // column
  //      if constexpr (coord == Coord::Row)
  //        std::swap(i, j);
  //
  //      // Get the global index of the `out` element
  //      GlobalElementIndex idx_el_gl_out(out_begin.row() + i, out_begin.col() + j);
  //      // Get the global index of the `in` element. If `coord == Coord::Row` use the permutation index on
  //      // the rows, otherwise use it on the columns.
  //      GlobalElementIndex idx_el_gl_in(in_offset + i, perm_arr[j]);
  //      if constexpr (coord == Coord::Row)
  //        idx_el_gl_in = GlobalElementIndex(perm_arr[i], in_offset + j);
  //
  //      // Get the index of the element local to the tile that contains it as well as the index of that
  //      // tile in the tiles arrays.
  //      TileElementIndex idx_el_out = distr.tileElementIndex(idx_el_gl_out);
  //      std::size_t idx_arr_out = to_sizet(distr.globalTileLinearIndex(idx_el_gl_out));
  //      TileElementIndex idx_el_in = distr.tileElementIndex(idx_el_gl_in);
  //      std::size_t idx_arr_in = to_sizet(distr.globalTileLinearIndex(idx_el_gl_in));
  //      //  copy from `in` to `out`
  //      out_tiles[idx_arr_out](idx_el_out) = in_tiles[idx_arr_in](idx_el_in);
  //    }
  //  });
}

}
