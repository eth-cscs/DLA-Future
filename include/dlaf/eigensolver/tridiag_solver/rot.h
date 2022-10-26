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

namespace wrapper {

template <class T>
void sendCol(comm::Communicator& comm, comm::IndexT_MPI rank_dest, comm::IndexT_MPI tag,
             const matrix::Tile<T, Device::CPU>& tile, TileElementIndex idx, SizeType n,
             MPI_Request* req) {
  DLAF_MPI_CHECK_ERROR(MPI_Isend(tile.ptr(idx), static_cast<int>(n), dlaf::comm::mpi_datatype<T>::type,
                                 rank_dest, tag, comm, req));
}
DLAF_MAKE_CALLABLE_OBJECT(sendCol);

template <class T>
void recvCol(comm::Communicator& comm, comm::IndexT_MPI rank_dest, comm::IndexT_MPI tag,
             const matrix::Tile<T, Device::CPU>& tile, SizeType n, MPI_Request* req) {
  DLAF_MPI_CHECK_ERROR(MPI_Irecv(tile.ptr({0, 0}), static_cast<int>(n),
                                 dlaf::comm::mpi_datatype<T>::type, rank_dest, tag, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(recvCol);

template <class T, class CommSender, class Sender>
auto scheduleSendCol(CommSender&& comm, comm::IndexT_MPI dest, comm::IndexT_MPI tag, Sender&& tile,
                     TileElementIndex idx, SizeType n) {
  return dlaf::internal::whenAllLift(std::forward<CommSender>(comm), dest, tag,
                                     std::forward<Sender>(tile), idx, n) |
         dlaf::comm::internal::transformMPI(sendCol_o);
}

template <class T, class CommSender, class Sender>
auto scheduleRecvCol(CommSender&& comm, comm::IndexT_MPI source, comm::IndexT_MPI tag, Sender&& tile,
                     SizeType n) {
  return dlaf::internal::whenAllLift(std::forward<CommSender>(comm), source, tag,
                                     std::forward<Sender>(tile), n) |
         dlaf::comm::internal::transformMPI(recvCol_o);
}

}

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

  const SizeType mb = dist.blockSize().rows();

  const matrix::Distribution dist_sub = [&]() {
    const SizeType sub_size = (i_last + 1 - i_begin) * mb;
    return matrix::Distribution({sub_size, sub_size}, dist.blockSize(), dist.commGridSize(),
                                dist.rankIndex(), dist.rankGlobalTile({i_begin, i_begin}));
  }();

  auto givens_rots_fn = [comm_row, mb, dist, dist_sub](std::vector<GivensRotation<T>> rots,
                                                       std::vector<matrix::Tile<T, D>> tiles,
                                                       matrix::Tile<T, D> tile_ws) {
    const SizeType m = dist_sub.localSize().rows();

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
      const auto& tile_x = [&]() -> matrix::Tile<T, D> const& {
        if (hasX) {
          const LocalTileIndex loc_tile{dist_sub.nextLocalTileFromGlobalElement<Coord::Row>(0),
                                        dist_sub.nextLocalTileFromGlobalElement<Coord::Col>(rot.i)};
          const std::size_t idx_x = to_sizet(dist_sub.localTileLinearIndex(loc_tile));
          return tiles[idx_x];
        }
        return tile_ws;
      }();
      const auto& tile_y = [&]() -> matrix::Tile<T, D> const& {
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

        auto tile = ex::just(std::cref(hasX ? tile_x : tile_y));
        const TileElementIndex idx = hasX ? idx_x : idx_y;

        cps.emplace_back(
            wrapper::scheduleSendCol<T>(comm_row, rank_partner, 0, std::move(tile), idx, m));
        cps.emplace_back(
            wrapper::scheduleRecvCol<T>(comm_row, rank_partner, 0, ex::just(std::cref(tile_ws)), m));
      }

      // each one computes his own, but just stores either x or y
      // (or both if are on the same rank)
      T* x = hasX ? tile_x.ptr(idx_x) : tile_ws.ptr({0, 0});
      T* y = hasY ? tile_y.ptr(idx_y) : tile_ws.ptr({0, 0});

      tt::sync_wait(di::whenAllLift(ex::when_all_vector(std::move(cps))) |
                    di::transform(di::Policy<DefaultBackend_v<D>>(), [rot, m, x, y](auto&&... ts) {
                      if constexpr (D == Device::CPU)
                        blas::rot(m, x, 1, y, 1, rot.c, rot.s);
                      // TODO GPU NOT IMPLEMENTED
                      // else
                      //   givensRotationOnDevice(m, x, y, rot.c, rot.s, ts...);
                    }));
    }
  };

  const matrix::Distribution dist_ws({dist.size().rows(), 1}, dist.blockSize(), dist.commGridSize(),
                                     dist.rankIndex(), dist.sourceRankIndex());
  // TODO allocate just sub-range
  matrix::Panel<Coord::Col, T, D> workspace(dist_ws);

  const TileCollector tc{i_begin, i_last};
  // TODO check if there could be any problem passing just the first tile of workspace (and using the full panel)
  di::whenAllLift(std::forward<GRSender>(rots_fut), ex::when_all_vector(tc.readwrite(mat)),
                  workspace.readwrite_sender({0, 0})) |
      di::transformDetach(di::Policy<Backend::MC>(), givens_rots_fn);
}
}
