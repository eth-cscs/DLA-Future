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
#include "dlaf/util_matrix.h"
#include "dlaf/types.h"

namespace dlaf::eigensolver::internal {

// Interleaves two intervals of length `l` split in blocks of size `b` starting at offsets `o1` and
// `o2` respectively and returns an array of indices where the splits have occured.
//
// o1
//  │
//  └► │   │   │   │   │
//    ─┴───┴───┴───┴───┴───  ◄─┐
//               ▲             │
// o2      b ─┬──┘              l
// │          │                │
// └─►  │   │ ▼ │   │   │      │
//    ──┴───┴───┴───┴───┴──  ◄─┘
//
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

// Matrix multiplications C = A * B
//
// ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
// │    *    *    *   │   │    *    *    *   │   │    *    *    *   │
// │   ┌─────────┐    │   │   ┌─────┐        │   │                  │
// │*  │ A       │    │   │*  │B    │        │   │*                 │
// │   │         │    │   │   │     │        │   │            ┌─────┤
// │   │         │    │ X │   │     │        │ = │            │C    │
// │*  │         │    │   │*  │     │        │   │*           │     │
// │   │         │    │   │   └─────┘        │   │            │     │
// │   └─────────┘    │   │                  │   │            │     │
// │*                 │   │*                 │   │*           │     │
// └──────────────────┘   └──────────────────┘   └────────────┴─────┘
//
template <class T>
void submatrixGEMM(SizeType len_m, SizeType len_n, SizeType len_k, GlobalElementIndex idx_a,
                   GlobalElementIndex idx_b, GlobalElementIndex idx_c, matrix::Distribution const& distr,
                   const std::vector<matrix::Tile<T, Device::CPU>>& tiles_a,
                   const std::vector<matrix::Tile<T, Device::CPU>>& tiles_b,
                   std::vector<matrix::Tile<T, Device::CPU>>& tiles_c) {
  DLAF_ASSERT(matrix::square_blocksize(distr), distr.blockSize());
  SizeType nb = distr.blockSize().rows();

  std::vector<SizeType> splits_m =
      interleaveSplits(len_m, nb, distr.distanceToAdjacentTile<Coord::Row>(idx_a.row()),
                       distr.distanceToAdjacentTile<Coord::Row>(idx_c.row()));
  std::vector<SizeType> splits_n =
      interleaveSplits(len_n, nb, distr.distanceToAdjacentTile<Coord::Col>(idx_b.col()),
                       distr.distanceToAdjacentTile<Coord::Col>(idx_c.col()));
  std::vector<SizeType> splits_k =
      interleaveSplits(len_k, nb, distr.distanceToAdjacentTile<Coord::Col>(idx_a.col()),
                       distr.distanceToAdjacentTile<Coord::Row>(idx_b.row()));

  // Constructs subtiles from `row_splits` and `col_splits`.
  //
  // A subtile is represented by triad (i_tile, i_subtile, sz_subtile):
  //
  // i_tile   i_subtile
  //    │       │
  //    └►┌─────┼─────┐
  //      │     │     │
  //      │     └►┌───┤
  //      │       │   │
  //      │       │   │
  //      │       │   │
  //      └───────┴───┘
  //
  for (std::size_t j = 0; j < splits_n.size() - 1; ++j) {
    // Iterate over interleaved column splits of `C` and `B`
    SizeType split_n = splits_n[j];
    SizeType len_subtile_n = splits_n[j + 1] - split_n;

    for (std::size_t i = 0; i < splits_m.size() - 1; ++i) {
      // Iterate over interleaved row splits of `C` and `A`
      SizeType split_m = splits_m[i];
      SizeType len_subtile_m = splits_m[i + 1] - split_m;

      // `c` subtile from (len_m x len_n)
      GlobalElementIndex i_split_gl_c = idx_c + GlobalElementSize(split_m, split_n);
      TileElementIndex i_subtile_c = distr.tileElementIndex(i_split_gl_c);
      auto& tile_c = tiles_c[to_sizet(distr.globalTileLinearIndex(i_split_gl_c))];

      for (std::size_t k = 0; k < splits_k.size() - 1; ++k) {
        // Iterate over interleaved column splits of `A` and row splits of `B`
        SizeType split_k = splits_k[k];
        SizeType len_subtile_k = splits_k[k + 1] - split_k;

        // `a` subtile from (len_m x len_k)
        GlobalElementIndex i_split_gl_a = idx_a + GlobalElementSize(split_m, split_k);
        TileElementIndex i_subtile_a = distr.tileElementIndex(i_split_gl_a);
        const auto& tile_a = tiles_a[to_sizet(distr.globalTileLinearIndex(i_split_gl_a))];

        // `b` subtile from (len_k x len_n)
        GlobalElementIndex i_split_gl_b = idx_b + GlobalElementSize(split_k, split_n);
        TileElementIndex i_subtile_b = distr.tileElementIndex(i_split_gl_b);
        const auto& tile_b = tiles_b[to_sizet(distr.globalTileLinearIndex(i_split_gl_b))];

        // beta = 1 accumulates results in `c` subtile from subtiles `a` and `b` tiles along the `k`
        // dimension
        //
        // beta = 0 is needed for the first subtile gemm as `C` may be non-zero
        T beta = 1;
        if (k == 0)
          beta = 0;

        // GEMM subtiles `c = a * b`
        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, len_subtile_m,
                   len_subtile_n, len_subtile_k, T(1), tile_a.ptr(i_subtile_a), tile_a.ld(),
                   tile_b.ptr(i_subtile_b), tile_b.ld(), beta, tile_c.ptr(i_subtile_c), tile_c.ld());
      }
    }
  }
}
}
