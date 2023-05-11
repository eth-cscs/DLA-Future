//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#ifdef DLAF_WITH_GPU
#include <whip.hpp>
#endif

#include "dlaf/common/assert.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/common/single_threaded_blas.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/datatypes.h"
#include "dlaf/communication/kernels/p2p.h"
#include "dlaf/eigensolver/tridiag_solver/kernels.h"
#include "dlaf/eigensolver/tridiag_solver/tile_collector.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"
#include "dlaf/sender/transform_mpi.h"
#include "dlaf/sender/when_all_lift.h"

namespace dlaf::eigensolver::internal {

namespace wrapper {

template <Device D, class T>
void sendCol(comm::Communicator& comm, comm::IndexT_MPI rank_dest, comm::IndexT_MPI tag,
             const T* col_data, const SizeType n, MPI_Request* req) {
  static_assert(D == Device::CPU, "This function works just with CPU memory.");

  DLAF_MPI_CHECK_ERROR(MPI_Isend(col_data, static_cast<int>(n), dlaf::comm::mpi_datatype<T>::type,
                                 rank_dest, tag, comm, req));
}

template <Device D, class T>
void recvCol(comm::Communicator& comm, comm::IndexT_MPI rank_dest, comm::IndexT_MPI tag, T* col_data,
             const SizeType n, MPI_Request* req) {
  static_assert(D == Device::CPU, "This function works just with CPU memory.");

  DLAF_MPI_CHECK_ERROR(MPI_Irecv(col_data, static_cast<int>(n), dlaf::comm::mpi_datatype<T>::type,
                                 rank_dest, tag, comm, req));
}

template <Device D, class T, class CommSender>
auto scheduleSendCol(CommSender&& comm, comm::IndexT_MPI dest, comm::IndexT_MPI tag, const T* col_data,
                     const SizeType n) {
  namespace di = dlaf::internal;

  if constexpr (D == Device::CPU) {
    return di::whenAllLift(std::forward<CommSender>(comm), dest, tag, col_data, n) |
           dlaf::comm::internal::transformMPI(sendCol<D, T>);
  }
#ifdef DLAF_WITH_GPU
  else if constexpr (D == Device::GPU) {
    namespace ex = pika::execution::experimental;
    using dlaf::matrix::internal::CopyBackend_v;
    using pika::execution::thread_priority;

    return ex::just(memory::MemoryView<T, Device::CPU>{n}) |
           ex::let_value([comm = std::forward<CommSender>(comm), dest, tag, col_data,
                          n](memory::MemoryView<T, Device::CPU>& mem_view) mutable {
             auto copy =
                 ex::just(mem_view(), col_data, to_sizet(n) * sizeof(T), whip::memcpy_device_to_host) |
                 di::transform(di::Policy<CopyBackend_v<Device::GPU, Device::CPU>>{thread_priority::high},
                               whip::memcpy_async);

             return di::whenAllLift(std::move(copy), std::forward<CommSender>(comm), dest, tag,
                                    mem_view(), n) |
                    dlaf::comm::internal::transformMPI(sendCol<Device::CPU, T>);
           });
  }
#endif
  else {
    DLAF_STATIC_UNIMPLEMENTED(T);
  }
}

template <Device D, class T, class CommSender>
auto scheduleRecvCol(CommSender&& comm, comm::IndexT_MPI source, comm::IndexT_MPI tag, T* col_data,
                     SizeType n) {
  namespace di = dlaf::internal;

  if constexpr (D == Device::CPU) {
    return di::whenAllLift(std::forward<CommSender>(comm), source, tag, col_data, n) |
           dlaf::comm::internal::transformMPI(recvCol<D, T>);
  }
#ifdef DLAF_WITH_GPU
  else if constexpr (D == Device::GPU) {
    using dlaf::matrix::internal::CopyBackend_v;
    using pika::execution::thread_priority;

    namespace ex = pika::execution::experimental;

    return ex::just(memory::MemoryView<T, Device::CPU>{n}) |
           ex::let_value([comm = std::forward<CommSender>(comm), source, tag, col_data,
                          n](memory::MemoryView<T, Device::CPU>& mem_view) mutable {
             auto recv = di::whenAllLift(std::forward<CommSender>(comm), source, tag, mem_view(), n) |
                         dlaf::comm::internal::transformMPI(recvCol<Device::CPU, T>);

             return di::whenAllLift(std::move(recv), col_data, mem_view(), to_sizet(n) * sizeof(T),
                                    whip::memcpy_host_to_device) |
                    di::transform(di::Policy<
                                      CopyBackend_v<Device::CPU, Device::GPU>>{thread_priority::high},
                                  whip::memcpy_async);
           });
  }
#endif
  else {
    DLAF_STATIC_UNIMPLEMENTED(T);
  }
}
}

template <class T>
struct GivensRotation {
  SizeType i;  // the first column index
  SizeType j;  // the second column index
  T c;         // cosine
  T s;         // sine
};

/// Apply GivenRotations to tiles of the distributed square sub-matrix identified by tile in range
/// [i_begin, i_last].
///
/// @param comm_row row communicator
/// @param tag is used for all communications happening over @p comm_row
/// @param i_begin global tile index for both row and column identifying the start of the sub-matrix
/// @param i_last global tile index for both row and column identifying the end of the sub-matrix
/// (inclusive)
/// @param rots_fut GivenRotations to apply (element column indices of rotations are relative to the
/// sub-matrix)
/// @param mat distributed matrix where the sub-matrix is located
///
/// @pre mat is distributed along rows the same way as comm_row
/// @pre memory layout of @p mat is column major.
template <class T, Device D, class GRSender>
void applyGivensRotationsToMatrixColumns(comm::Communicator comm_row, comm::IndexT_MPI tag,
                                         SizeType i_begin, SizeType i_last, GRSender&& rots_fut,
                                         Matrix<T, D>& mat) {
  // Note:
  // a column index may be paired to more than one other index, this may lead to a race
  // condition if parallelized trivially. Current implementation is serial.

  namespace ex = pika::execution::experimental;
  namespace tt = pika::this_thread::experimental;
  namespace di = dlaf::internal;

  DLAF_ASSERT_HEAVY(comm_row.size() == mat.commGridSize().cols(), comm_row.size(),
                    mat.commGridSize().cols());
  DLAF_ASSERT_HEAVY(comm_row.rank() == mat.rankIndex().col(), comm_row.rank(), mat.rankIndex().col());

  const matrix::Distribution& dist = mat.distribution();

  const SizeType mb = dist.blockSize().rows();
  const SizeType i_end = i_last + 1;
  const SizeType range_size_limit = std::min(dist.size().rows(), i_end * mb);
  const SizeType range_size = range_size_limit - i_begin * mb;

  // Note:
  // Some ranks might not participate to the application of given rotations. This logic checks which
  // ranks are involved in order to operate in the range [i_begin, i_last].
  const bool isInRangeRow = [&]() {
    const SizeType begin = dist.nextLocalTileFromGlobalTile<Coord::Row>(i_begin);
    const SizeType end = dist.nextLocalTileFromGlobalTile<Coord::Row>(i_end);
    return end - begin != 0;
  }();

  const bool isInRangeCol = [&]() {
    const SizeType begin = dist.nextLocalTileFromGlobalTile<Coord::Col>(i_begin);
    const SizeType end = dist.nextLocalTileFromGlobalTile<Coord::Col>(i_end);
    return end - begin != 0;
  }();

  const bool isInRange = isInRangeRow && isInRangeCol;
  if (!isInRange)
    return;

  // Note:
  // This is the distribution that will be used inside the task. Differently from the original one,
  // this targets just the range defined by [i_begin, i_last], but keeping the same distribution over
  // ranks.
  const matrix::Distribution dist_sub({range_size, range_size}, dist.blockSize(), dist.commGridSize(),
                                      dist.rankIndex(), dist.rankGlobalTile({i_begin, i_begin}));

  auto givens_rots_fn = [comm_row, tag, dist_sub, mb](std::vector<GivensRotation<T>> rots,
                                                      std::vector<matrix::Tile<T, D>> tiles,
                                                      std::vector<matrix::Tile<T, D>> all_ws) {
    // Note:
    // It would have been enough to just get the first tile from the beginning, and it would have
    // worked anyway (thanks to the fact that panel has its own memorychunk and the first tile would
    // keep alive the entire chunk, so also the part of other tiles, through its memory view).
    // Anyway, it is more clean to get them all, and then just here use a single one to access the
    // full column.
    matrix::Tile<T, D>& tile_ws = all_ws[0];

    // Note:
    // The entire algorithm relies on a strong assumption about memory layout of all tiles involved,
    // i.e. both the input tiles selected from the matrix and the tiles for the workspace.
    // By relying on the fact that all of them are stored in a column-layout, we can work on column
    // vectors ignoring their distribution over different tiles, and it would be enough just getting
    // the pointer to the top head of the column and all other data can be easily accessed since it
    // is stored contiguously after that.
    // Ignoring tile organization of the memory comes handy also for communication. Indeed, during
    // column exchange, just a single MPI operation per column is issued, instead of communicating
    // independently the parts of column from each tile.
    auto getColPtr = [&dist_sub, &tiles, &tile_ws](const SizeType col_index, const bool hasIt) -> T* {
      if (hasIt) {
        const LocalTileIndex tile_col{dist_sub.nextLocalTileFromGlobalElement<Coord::Row>(0),
                                      dist_sub.nextLocalTileFromGlobalElement<Coord::Col>(col_index)};
        const std::size_t linear_tile_col = to_sizet(dist_sub.localTileLinearIndex(tile_col));
        return tiles[linear_tile_col].ptr(dist_sub.tileElementIndex({0, col_index}));
      }
      return tile_ws.ptr({0, 0});
    };

    const SizeType m = dist_sub.localSize().rows();

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

      std::vector<ex::unique_any_sender<>> comm_checkpoints;
      if (!hasBothXY) {
        const comm::IndexT_MPI rank_partner = hasX ? rankColY : rankColX;

        const T* col_send = hasX ? col_x : col_y;
        T* col_recv = hasX ? col_y : col_x;

        // Note:
        // These communications use raw pointers, so correct lifetime management of related tiles
        // is up to the caller.
        comm_checkpoints.emplace_back(
            wrapper::scheduleSendCol<D, T>(comm_row, rank_partner, tag, col_send, m));
        comm_checkpoints.emplace_back(
            wrapper::scheduleRecvCol<D, T>(comm_row, rank_partner, tag, col_recv, m));
      }

      // Note:
      // With a single workspace, just one rotation per time can be done.
      //
      // Communications of a step can happen together, since the only available workspace is used
      // only for receiving the counterpart column. When communications are finished, the rank has
      // everything needed to compute the rotation, so the actual computation task can start.
      //
      // However, communications of multiple steps, even with multiple workspaces, might need to be
      // kept serialized. Indeed, if rotations are not independent, i.e. if the same column appears
      // in multiple rotations, the second time the column is communicated, it must be the one rotated
      // resulting from the previous step, not the original one.
      // For simplicity, we add the constraint that communications of a step depends on previous step
      // computation, which might be too tight, but currently it looks like the most straightforward
      // solution.
      //
      // Moreover, even if rotations are independent, scheduling all communications all together
      // would require a tag ad hoc ensuring that communication between same ranks do not get mixed
      // (in addition to having a tag ensuring that other calls to this algorithm do not get mixed too).
      //
      // Current serialization of the algorithm ensures that communications cannot be scheduled all
      // together beforehand, and with them, also temporary buffers for the GPU case. If this assumption
      // will drop, it is relevant to highlight that there is nothing that would stop to schedule and,
      // more importantly, allocate all of them together.
      tt::sync_wait(
          ex::when_all_vector(std::move(comm_checkpoints)) |
          di::transform(di::Policy<DefaultBackend_v<D>>(), [rot, m, col_x, col_y](auto&&... ts) {
            // Note:
            // each one computes his own, but just stores either x or y (or both if on the same rank)
            if constexpr (D == Device::CPU) {
              static_assert(sizeof...(ts) == 0, "Parameter pack should be empty for MC.");
              dlaf::common::internal::SingleThreadedBlasScope single;
              blas::rot(m, col_x, 1, col_y, 1, rot.c, rot.s);
            }
#ifdef DLAF_WITH_GPU
            else if constexpr (D == Device::GPU) {
              givensRotationOnDevice(m, col_x, col_y, rot.c, rot.s, ts...);
            }
#endif
            else {
              DLAF_STATIC_UNIMPLEMENTED(T);
            }
          }));
    }
  };

  // Note:
  // Using a combination of shrinked distribution and an offset given to the panel, the workspace
  // is allocated just for the part strictly needed by the range [i_begin, i_last]
  const matrix::Distribution dist_ws({range_size_limit, 1}, dist.blockSize(), dist.commGridSize(),
                                     dist.rankIndex(), dist.sourceRankIndex());
  matrix::Panel<Coord::Col, T, D> workspace(dist_ws, GlobalTileIndex(i_begin, i_begin));

  const TileCollector tc(i_begin, i_last);

  ex::when_all(std::forward<GRSender>(rots_fut), ex::when_all_vector(tc.readwrite(mat)),
               ex::when_all_vector(select(workspace, workspace.iteratorLocal()))) |
      di::transformDetach(di::Policy<Backend::MC>(), givens_rots_fn);
}
}
