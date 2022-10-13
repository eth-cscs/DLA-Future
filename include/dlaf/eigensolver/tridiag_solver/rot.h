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
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/kernels/p2p.h"
#include "dlaf/eigensolver/tridiag_solver/merge.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/sender/transform_mpi.h"
#include "dlaf/sender/when_all_lift.h"

namespace dlaf::eigensolver::internal {

// @param tiles The tiles of the matrix between tile indices `(i_begin, i_begin)` and `(i_end, i_end)`
// that are potentially affected by the Givens rotations.
// @param n column size
//
// @pre the memory layout of the matrix from which the tiles are coming is column major.
//
// Note: a column index may be paired to more than one other index, this may lead to a race condition if
//       parallelized trivially. Current implementation is serial.
template <class T, Device D, class GRSender>
void applyGivensRotationsToMatrixColumns(comm::Communicator comm_row, SizeType i_begin, SizeType i_last,
                                         GRSender&& rots_fut, Matrix<T, D>& mat) {
  namespace ex = pika::execution::experimental;
  namespace tt = pika::this_thread::experimental;
  namespace di = dlaf::internal;

  const matrix::Distribution& dist = mat.distribution();

  const SizeType mb = mat.distribution().blockSize().rows();

  const matrix::Distribution dist_sub = [=]() {
    const SizeType size = (i_last + 1 - i_begin) * mb;
    return matrix::Distribution({size, size}, dist.blockSize(), dist.commGridSize(), dist.rankIndex(),
                                dist.rankGlobalTile({i_begin, i_begin}));
  }();

  const SizeType n = dist_sub.localSize().rows();

  auto givens_rots_fn = [comm_row, n, mb, dist, dist_sub](const std::vector<GivensRotation<T>>& rots,
                                                          auto&& tiles, [[maybe_unused]] auto&&... ts) {
    const matrix::Distribution dist_ws({dist.size().rows(), 1}, dist.blockSize(), dist.commGridSize(),
                                       dist.rankIndex(), dist.sourceRankIndex());
    // TODO allocate just sub-range
    matrix::Panel<Coord::Col, T, D> workspace(dist_ws);

    for (const GivensRotation<T>& rot : rots) {
      const SizeType j_x = rot.i / mb;
      const SizeType j_y = rot.j / mb;

      const comm::IndexT_MPI rankColX = dist_sub.template rankGlobalTile<Coord::Col>(j_x);
      const comm::IndexT_MPI rankColY = dist_sub.template rankGlobalTile<Coord::Col>(j_y);

      const bool hasX = dist.rankIndex().col() == rankColX;
      const bool hasY = dist.rankIndex().col() == rankColY;

      if (!hasX && !hasY)
        continue;

      // TODO assume column layout and operate on the full column? also workspace must obey
      auto&& tile_ws = workspace({0, 0}).get();
      const auto& tile_x = [&]() -> matrix::Tile<T, Device::CPU> const& {
        if (hasX) {
          const LocalTileIndex loc_tile{dist_sub.nextLocalTileFromGlobalElement<Coord::Row>(0),
                                        dist_sub.nextLocalTileFromGlobalElement<Coord::Col>(rot.i)};
          const std::size_t idx_x = to_sizet(dist_sub.localTileLinearIndex(loc_tile));
          return tiles[idx_x];
        }
        return tile_ws;
      }();
      const auto& tile_y = [&]() -> matrix::Tile<T, Device::CPU> const& {
        if (hasY) {
          const LocalTileIndex loc_tile{dist_sub.nextLocalTileFromGlobalElement<Coord::Row>(0),
                                        dist_sub.nextLocalTileFromGlobalElement<Coord::Col>(rot.j)};
          const std::size_t idx_y = to_sizet(dist_sub.localTileLinearIndex(loc_tile));
          return tiles[idx_y];
        }
        return tile_ws;
      }();

      const bool hasBothXY = hasX && hasY;

      const TileElementIndex idx_x(0, rot.i % mb);  // TODO check if dist can be used
      const TileElementIndex idx_y(0, rot.j % mb);

      std::vector<ex::unique_any_sender<>> cps;
      if (!hasBothXY) {
        // TODO compute TAG

        const comm::IndexT_MPI rank_partner = hasX ? rankColY : rankColX;

        // TODO possible optimization, check if it is zero or not

        auto sender = [idx = hasX ? idx_x : idx_y, n](comm::Communicator& comm,
                                                      comm::IndexT_MPI rank_dest, comm::IndexT_MPI tag,
                                                      const auto& tile, MPI_Request* req) {
          DLAF_MPI_CHECK_ERROR(MPI_Isend(tile.ptr(idx), static_cast<int>(n),
                                         dlaf::comm::mpi_datatype<T>::type, rank_dest, tag, comm, req));
        };
        auto cp_send =
            dlaf::internal::whenAllLift(comm_row, rank_partner, 0, std::cref(hasX ? tile_x : tile_y)) |
            dlaf::comm::internal::transformMPI(sender);

        auto receiver = [n](comm::Communicator& comm, comm::IndexT_MPI rank_dest, comm::IndexT_MPI tag,
                            const matrix::Tile<T, Device::CPU>& tile, MPI_Request* req) {
          DLAF_MPI_CHECK_ERROR(MPI_Irecv(tile.ptr({0, 0}), static_cast<int>(n),
                                         dlaf::comm::mpi_datatype<T>::type, rank_dest, tag, comm, req));
        };
        auto cp_recv = dlaf::internal::whenAllLift(comm_row, rank_partner, 0, std::cref(tile_ws)) |
                       dlaf::comm::internal::transformMPI(receiver);
        cps.emplace_back(std::move(cp_send));
        cps.emplace_back(std::move(cp_recv));
      }

      // each one computes his own, but just stores either x or y
      // (or both if are on the same rank)
      T* x = hasX ? tile_x.ptr(idx_x) : tile_ws.ptr({0, 0});
      T* y = hasY ? tile_y.ptr(idx_y) : tile_ws.ptr({0, 0});

      tt::sync_wait(ex::when_all_vector(std::move(cps)) |
                    ex::then([rot, n, x, y]() { blas::rot(n, x, 1, y, 1, rot.c, rot.s); }));
    }
  };

  TileCollector tc{i_begin, i_last};
  auto sender =
      di::whenAllLift(std::forward<GRSender>(rots_fut), ex::when_all_vector(tc.readwrite(mat)));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(), givens_rots_fn, std::move(sender));
}
}
