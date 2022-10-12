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
#include "dlaf/common/range2d.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/kernels/p2p.h"
#include "dlaf/eigensolver/tridiag_solver/merge.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/sender/when_all_lift.h"

namespace dlaf::eigensolver::internal {

// Assumption: the memory layout of the matrix from which the tiles are coming is column major.
//
// @param tiles The tiles of the matrix between tile indices `(i_begin, i_begin)` and `(i_end, i_end)`
// that are potentially affected by the Givens rotations.
// @param n column size
//
// Note: a column index may be paired to more than one other index, this may lead to a race condition if
//       parallelized trivially. Current implementation is serial.
template <class T, Device D, class GRSender>
void applyGivensRotationsToMatrixColumns(comm::CommunicatorGrid grid, SizeType i_begin, SizeType i_last,
                                         GRSender&& rots_fut, Matrix<T, D>& mat) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  // TODO grid pipeline?

  // const SizeType m = problemSize(i_begin, i_last, mat.distribution());
  const SizeType mb = mat.distribution().blockSize().rows();

  const matrix::Distribution& dist = mat.distribution();

  auto givens_rots_fn =
      [mb, dist,
       i_begin](comm::CommunicatorGrid grid, const std::vector<GivensRotation<T>>& rots,
                // const std::vector<std::pair<LocalTileIndex, matrix::Tile<T, D>>>& tiles_with_indices,
                auto tile_indices, auto&& tiles, [[maybe_unused]] auto&&... ts) {
        // TODO replace this with implementation
        dlaf::internal::silenceUnusedWarningFor(grid, rots, tiles, ts...);

        // TODO workspace
        constexpr SizeType n_workspaces = 2;
        common::RoundRobin<matrix::Panel<Coord::Col, T, D>> workspace(n_workspaces, dist);

        // Distribution of the merged subproblems
        // matrix::Distribution distr(LocalElementSize(n, n), TileElementSize(nb, nb));

        for (const GivensRotation<T>& rot : rots) {
          // TODO compute if this rank is in a x or y column
          const SizeType j_x = i_begin + rot.i / mb;
          const SizeType j_y = i_begin + rot.j / mb;

          // TODO assume column layout and operate on the full column? also workspace must obey

          // local, just on columns of x/y
          for (std::size_t index = 0; index < tile_indices.size(); ++index) {
            auto&& tile = tiles[index];
            const LocalTileIndex ij(tile_indices[index]);
            const SizeType j = dist.template globalTileFromLocalTile<Coord::Col>(ij.col());

            const bool hasX = j == j_x;
            const bool hasY = j == j_y;

            if (j != j_x && j != j_y)
              continue;

            const comm::IndexT_MPI rankColX = dist.template rankGlobalTile<Coord::Col>(j_x);
            const comm::IndexT_MPI rankColY = dist.template rankGlobalTile<Coord::Col>(j_y);

            // Get the index of the tile that has column `rot.i` and the the index of the column
            // within the tile.

            // SizeType i_tile = distr.globalTileLinearIndex(GlobalElementIndex(0, rot.i));
            // SizeType i_el = distr.tileElementFromGlobalElement<Coord::Col>(rot.i);
            // T* x = tiles[to_sizet(i_tile)].ptr(TileElementIndex(0, i_el));

            // Get the index of the tile that has column `rot.j` and the the index of the column
            // within the tile.

            // SizeType j_tile = distr.globalTileLinearIndex(GlobalElementIndex(0, rot.j));
            // SizeType j_el = distr.tileElementFromGlobalElement<Coord::Col>(rot.j);
            // T* y = tiles[to_sizet(j_tile)].ptr(TileElementIndex(0, j_el));

            const bool hasBothXY = hasX && hasY;
            if (hasBothXY) {
              // no communication, they all have everything locally
            }
            else {
              const comm::IndexT_MPI rank_partner = hasX ? rankColY : rankColX;

              // TODO one is sent, the other is received
              // TODO splitTile destination + get right workspace
              // TODO possible optimization, check if it is zero or not
              // comm::scheduleSend(CommSender && pcomm, IndexT_MPI dest, IndexT_MPI tag, Sender && tile);
              // comm::scheduleRecv(CommSender && pcomm, IndexT_MPI dest, IndexT_MPI tag, Sender && tile);
            }

            // TODO each one computes his own, but just stores either x or y (or both if are on the
            // same rank)

            if (hasBothXY) {
              // TODO check if dist can be used
              T* x = tile.ptr(TileElementIndex(0, rot.i % mb));
              T* y = tile.ptr(TileElementIndex(0, rot.j % mb));

              // Apply Givens rotations
              if constexpr (D == Device::CPU) {
                blas::rot(mb, x, 1, y, 1, rot.c, rot.s);
              }
              else {
                givensRotationOnDevice(mb, x, y, rot.c, rot.s, ts...);
              }
            }
          }
        }
      };

  const TileCollector tc{i_begin, i_last};
  const auto [begin, size] = tc.iteratorLocal(mat.distribution());
  const auto range = common::iterate_range2d(begin, size);
  auto sender =
      di::whenAllLift(grid, std::forward<GRSender>(rots_fut), std::vector(range.begin(), range.end()),
                      ex::when_all_vector(tc.readwrite(mat)));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(), givens_rots_fn, std::move(sender));
}

}
