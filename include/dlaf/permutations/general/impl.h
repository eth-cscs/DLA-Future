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

#include "dlaf/permutations/general/api.h"

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/lapack/gpu/lacpy.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/schedulers.h"
#include "dlaf/sender/transform.h"
#include "dlaf/sender/when_all_lift.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

#include "pika/algorithm.hpp"

namespace dlaf::permutations::internal {

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
// Note: `in_tiles` should be `const T` but to avoid extra allocations necessary for unwrapping
//       `std::vector<shared_future<matrix::Tile<const T, D>>>` it is left as non-const
template <class T, Coord coord>
void applyPermutations(GlobalElementIndex out_begin, GlobalElementSize sz, SizeType in_offset,
                       const matrix::Distribution& distr, const SizeType* perm_arr,
                       const std::vector<matrix::Tile<T, Device::CPU>>& in_tiles,
                       const std::vector<matrix::Tile<T, Device::CPU>>& out_tiles) {
  constexpr Coord orth_coord = orthogonal(coord);
  std::vector<SizeType> splits =
      util::interleaveSplits(sz.get<orth_coord>(), distr.blockSize().get<orth_coord>(),
                             distr.distanceToAdjacentTile<orth_coord>(in_offset),
                             distr.distanceToAdjacentTile<orth_coord>(out_begin.get<orth_coord>()));

  // Parallelized over the number of permuted columns or rows
  pika::for_loop(pika::execution::par, to_sizet(0), to_sizet(sz.get<coord>()), [&](SizeType i_perm) {
    for (std::size_t i_split = 0; i_split < splits.size() - 1; ++i_split) {
      SizeType split = splits[i_split];

      GlobalElementIndex i_split_gl_in(split + in_offset, perm_arr[i_perm]);
      GlobalElementIndex i_split_gl_out(split + out_begin.get<orth_coord>(),
                                        out_begin.get<coord>() + i_perm);
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

      dlaf::tile::lacpy<T>(region, i_subtile_in, tile_in, i_subtile_out, tile_out);
    }
  });
}

#if defined(DLAF_WITH_CUDA)
template <class T, Coord coord>
void applyPermutations(GlobalElementIndex out_begin, GlobalElementSize sz, SizeType in_offset,
                       const matrix::Distribution& distr, const SizeType* perm_arr,
                       const std::vector<matrix::Tile<T, Device::GPU>>& in_tiles,
                       const std::vector<matrix::Tile<T, Device::GPU>>& out_tiles, cudaStream_t stream) {
  constexpr Coord orth_coord = orthogonal(coord);
  std::vector<SizeType> splits =
      util::interleaveSplits(sz.get<orth_coord>(), distr.blockSize().get<orth_coord>(),
                             distr.distanceToAdjacentTile<orth_coord>(in_offset),
                             distr.distanceToAdjacentTile<orth_coord>(out_begin.get<orth_coord>()));

  for (SizeType i_perm = 0; i_perm < sz.get<coord>(); ++i_perm) {
    for (std::size_t i_split = 0; i_split < splits.size() - 1; ++i_split) {
      SizeType split = splits[i_split];

      GlobalElementIndex i_split_gl_in(split + in_offset, perm_arr[i_perm]);
      GlobalElementIndex i_split_gl_out(split + out_begin.get<orth_coord>(),
                                        out_begin.get<coord>() + i_perm);
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

      dlaf::gpulapack::lacpy<T>(blas::Uplo::General, region.rows(), region.cols(),
                                tile_in.ptr(i_subtile_in), tile_in.ld(), tile_out.ptr(i_subtile_out),
                                tile_out.ld(), stream);
    }
  };
}
#endif

template <Backend B, Device D, class T, Coord C>
void Permutations<B, D, T, C>::call(SizeType i_begin, SizeType i_end,
                                    Matrix<const SizeType, Device::CPU>& perms, Matrix<T, D>& mat_in,
                                    Matrix<T, D>& mat_out) {
  namespace ut = matrix::util;
  namespace ex = pika::execution::experimental;

  const matrix::Distribution& distr = mat_in.distribution();
  TileElementSize sz_last_tile = distr.tileSize(GlobalTileIndex(i_end, i_end));
  SizeType m = distr.globalTileElementDistance<Coord::Row>(i_begin, i_end) + sz_last_tile.rows();
  SizeType n = distr.globalTileElementDistance<Coord::Col>(i_begin, i_end) + sz_last_tile.cols();
  matrix::Distribution subm_distr(LocalElementSize(m, n), distr.blockSize());
  SizeType ntiles = i_end - i_begin + 1;

  auto sender = ex::when_all(ex::when_all_vector(ut::collectReadTiles(GlobalTileIndex(i_begin, 0),
                                                                      GlobalTileSize(ntiles, 1), perms)),
                             ex::when_all_vector(
                                 ut::collectReadWriteTiles(GlobalTileIndex(i_begin, i_begin),
                                                           GlobalTileSize(ntiles, ntiles), mat_in)),
                             ex::when_all_vector(
                                 ut::collectReadWriteTiles(GlobalTileIndex(i_begin, i_begin),
                                                           GlobalTileSize(ntiles, ntiles), mat_out)));

  auto permute_fn = [subm_distr](const auto& index_tile_futs, const auto& mat_in_tiles,
                                 const auto& mat_out_tiles, auto&&... ts) {
    TileElementIndex zero(0, 0);
    const SizeType* i_ptr = index_tile_futs[0].get().ptr(zero);
    applyPermutations<T, C>(GlobalElementIndex(0, 0), subm_distr.size(), 0, subm_distr, i_ptr,
                            mat_in_tiles, mat_out_tiles, std::forward<decltype(ts)>(ts)...);
  };
  ex::start_detached(dlaf::internal::transform<false>(dlaf::internal::Policy<B>(), std::move(permute_fn),
                                                      std::move(sender)));
}
}
