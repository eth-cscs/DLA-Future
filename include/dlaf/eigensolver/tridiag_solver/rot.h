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

#include "dlaf/common/assert.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/eigensolver/tridiag_solver/merge.h"
#include "dlaf/sender/when_all_lift.h"

namespace dlaf::eigensolver::internal {

// Assumption: the memory layout of the matrix from which the tiles are coming is column major.
//
// `tiles`: The tiles of the matrix between tile indices `(i_begin, i_begin)` and `(i_end, i_end)` that
// are potentially affected by the Givens rotations. `n` : column size
//
// Note: a column index may be paired to more than one other index, this may lead to a race condition if
//       parallelized trivially. Current implementation is serial.
//
template <class T, Device D>
void applyGivensRotationsToMatrixColumns(comm::CommunicatorGrid grid, SizeType i_begin, SizeType i_last,
                                         pika::future<std::vector<GivensRotation<T>>> rots_fut,
                                         Matrix<T, D>& mat) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  SizeType n = problemSize(i_begin, i_last, mat.distribution());
  SizeType nb = mat.distribution().blockSize().rows();

  auto givens_rots_fn = [n, nb](comm::CommunicatorGrid grid, const std::vector<GivensRotation<T>>& rots,
                                const std::vector<matrix::Tile<T, D>>& tiles,
                                [[maybe_unused]] auto&&... ts) {
    // TODO replace this with implementation
    dlaf::internal::silenceUnusedWarningFor(grid, rots, tiles, ts...);

    //   // Distribution of the merged subproblems
    //   matrix::Distribution distr(LocalElementSize(n, n), TileElementSize(nb, nb));

    for (const GivensRotation<T>& rot : rots) {
      //     // Get the index of the tile that has column `rot.i` and the the index of the column within
      //     the tile. SizeType i_tile = distr.globalTileLinearIndex(GlobalElementIndex(0, rot.i));
      //     SizeType i_el = distr.tileElementFromGlobalElement<Coord::Col>(rot.i); T* x =
      //     tiles[to_sizet(i_tile)].ptr(TileElementIndex(0, i_el));

      //     // Get the index of the tile that has column `rot.j` and the the index of the column within
      //     the tile. SizeType j_tile = distr.globalTileLinearIndex(GlobalElementIndex(0, rot.j));
      //     SizeType j_el = distr.tileElementFromGlobalElement<Coord::Col>(rot.j); T* y =
      //     tiles[to_sizet(j_tile)].ptr(TileElementIndex(0, j_el));

      //     // Apply Givens rotations
      //     if constexpr (D == Device::CPU) {
      //       blas::rot(n, x, 1, y, 1, rot.c, rot.s);
      //     }
      //     else {
      //       givensRotationOnDevice(n, x, y, rot.c, rot.s, ts...);
      //     }
    }
  };

  TileCollector tc{i_begin, i_last};

  auto sender = di::whenAllLift(grid, std::move(rots_fut), ex::when_all_vector(tc.readwrite(mat)));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(), std::move(givens_rots_fn), std::move(sender));
}

}
