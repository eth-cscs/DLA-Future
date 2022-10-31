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

template <Device D, class T>
void sendCol(comm::Communicator& comm, comm::IndexT_MPI rank_dest, comm::IndexT_MPI tag,
             const T* col_data, const SizeType n, MPI_Request* req) {
  if constexpr (D == Device::CPU) {
    DLAF_MPI_CHECK_ERROR(MPI_Isend(col_data, static_cast<int>(n), dlaf::comm::mpi_datatype<T>::type,
                                   rank_dest, tag, comm, req));
  }
  else {
    dlaf::internal::silenceUnusedWarningFor(comm, rank_dest, tag, col_data, n, req);
    DLAF_STATIC_UNIMPLEMENTED(T);
  }
}

template <Device D, class T>
void recvCol(comm::Communicator& comm, comm::IndexT_MPI rank_dest, comm::IndexT_MPI tag, T* col_data,
             const SizeType n, MPI_Request* req) {
  if constexpr (D == Device::CPU) {
    DLAF_MPI_CHECK_ERROR(MPI_Irecv(col_data, static_cast<int>(n), dlaf::comm::mpi_datatype<T>::type,
                                   rank_dest, tag, comm, req));
  }
  else {
    dlaf::internal::silenceUnusedWarningFor(comm, rank_dest, tag, col_data, n, req);
    DLAF_STATIC_UNIMPLEMENTED(T);
  }
}

template <Device D, class T, class CommSender>
auto scheduleSendCol(CommSender&& comm, comm::IndexT_MPI dest, comm::IndexT_MPI tag, const T* col_data,
                     const SizeType n) {
  return dlaf::internal::whenAllLift(std::forward<CommSender>(comm), dest, tag, col_data, n) |
         dlaf::comm::internal::transformMPI(sendCol<D, T>);
}

template <Device D, class T, class CommSender>
auto scheduleRecvCol(CommSender&& comm, comm::IndexT_MPI source, comm::IndexT_MPI tag, T* col_data,
                     SizeType n) {
  return dlaf::internal::whenAllLift(std::forward<CommSender>(comm), source, tag, col_data, n) |
         dlaf::comm::internal::transformMPI(recvCol<D, T>);
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

  const SizeType sub_square_edge = (i_last + 1 - i_begin) * mb;
  const GlobalElementSize sub_square_size(sub_square_edge, sub_square_edge);
  const matrix::Distribution dist_sub(sub_square_size, dist.blockSize(), dist.commGridSize(),
                                      dist.rankIndex(), dist.rankGlobalTile({i_begin, i_begin}));

  // TODO check workspace distribution (partial allocation and its usage with dist_sub)
  const matrix::Distribution dist_ws({dist.size().rows(), 1}, dist.blockSize(), dist.commGridSize(),
                                     dist.rankIndex(), dist.sourceRankIndex());
  // TODO allocate just sub-range
  matrix::Panel<Coord::Col, T, D> workspace(dist_ws);

  auto givens_rots_fn = [comm_row, dist_sub, mb](std::vector<GivensRotation<T>> rots,
                                                 std::vector<matrix::Tile<T, D>> tiles,
                                                 matrix::Tile<T, D> tile_ws) {
    const SizeType m = dist_sub.localSize().rows();

    auto getColPtr = [&dist_sub, &tiles, &tile_ws](const SizeType col_index, const bool hasIt) -> T* {
      // TODO document column layout assumption
      // TODO assume column layout and operate on the full column? also workspace must obey
      if (hasIt) {
        const LocalTileIndex tile_col{dist_sub.nextLocalTileFromGlobalElement<Coord::Row>(0),
                                      dist_sub.nextLocalTileFromGlobalElement<Coord::Col>(col_index)};
        const std::size_t linear_tile_col = to_sizet(dist_sub.localTileLinearIndex(tile_col));
        return tiles[linear_tile_col].ptr(dist_sub.tileElementIndex({0, col_index}));
      }
      return tile_ws.ptr({0, 0});
    };

    ex::unique_any_sender<> serializer = ex::just();
    for (const GivensRotation<T>& rot : rots) {
      const SizeType j_x = rot.i / mb;
      const SizeType j_y = rot.j / mb;

      const comm::IndexT_MPI rankColX = dist_sub.template rankGlobalTile<Coord::Col>(j_x);
      const comm::IndexT_MPI rankColY = dist_sub.template rankGlobalTile<Coord::Col>(j_y);

      const bool hasX = dist_sub.rankIndex().col() == rankColX;
      const bool hasY = dist_sub.rankIndex().col() == rankColY;

      if (!hasX && !hasY)
        continue;

      T* col_x = getColPtr(rot.i, hasX);
      T* col_y = getColPtr(rot.j, hasY);

      const bool hasBothXY = hasX && hasY;

      std::vector<ex::unique_any_sender<>> cps;
      if (!hasBothXY) {
        // TODO compute TAG

        const comm::IndexT_MPI rank_partner = hasX ? rankColY : rankColX;

        // TODO possible optimization, check if it is zero or not

        const T* col_send = hasX ? col_x : col_y;
        T* col_recv = hasX ? col_y : col_x;

        cps.emplace_back(wrapper::scheduleSendCol<D, T>(comm_row, rank_partner, 0, col_send, m));
        cps.emplace_back(wrapper::scheduleRecvCol<D, T>(comm_row, rank_partner, 0, col_recv, m));
      }

      // each one computes his own, but just stores either x or y
      // (or both if are on the same rank)
      serializer =
          di::whenAllLift(std::move(serializer), ex::when_all_vector(std::move(cps))) |
          di::transform(di::Policy<DefaultBackend_v<D>>(), [rot, m, col_x, col_y](auto&&... ts) {
            if constexpr (D == Device::CPU)
              blas::rot(m, col_x, 1, col_y, 1, rot.c, rot.s);
            // TODO GPU NOT IMPLEMENTED
            // else
            //   givensRotationOnDevice(m, x, y, rot.c, rot.s, ts...);
          });
    }
    tt::sync_wait(std::move(serializer));
  };

  const TileCollector tc{i_begin, i_last};

  // TODO check if there could be any problem passing just the first tile of workspace (and using the full panel)
  di::whenAllLift(std::forward<GRSender>(rots_fut), ex::when_all_vector(tc.readwrite(mat)),
                  workspace.readwrite_sender({0, 0})) |
      di::transformDetach(di::Policy<Backend::MC>(), givens_rots_fn);
}
}
