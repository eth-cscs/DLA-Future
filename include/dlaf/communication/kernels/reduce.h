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

#ifdef DLAF_WITH_CUDA_RDMA
#warning "Reduce is not using CUDA_RDMA."
#endif

/// @file

#include <mpi.h>

#include <pika/execution.hpp>

#include "dlaf/common/data.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/schedulers.h"
#include "dlaf/sender/traits.h"
#include "dlaf/sender/transform_mpi.h"
#include "dlaf/sender/with_temporary_tile.h"

namespace dlaf::comm {
namespace internal {
template <class T, Device D>
void reduceRecvInPlace(common::PromiseGuard<comm::Communicator> pcomm, MPI_Op reduce_op,
                       const matrix::Tile<T, D>& tile, MPI_Request* req) {
  DLAF_ASSERT(tile.is_contiguous(), "");

  auto& comm = pcomm.ref();
  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(MPI_Ireduce(MPI_IN_PLACE, msg.data(), msg.count(), msg.mpi_type(), reduce_op,
                                   comm.rank(), comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(reduceRecvInPlace);

template <class T, Device D>
void reduceSend(common::PromiseGuard<comm::Communicator> pcomm, comm::IndexT_MPI rank_root,
                MPI_Op reduce_op, const matrix::Tile<const T, D>& tile, MPI_Request* req) {
  DLAF_ASSERT(tile.is_contiguous(), "");

  auto& comm = pcomm.ref();
  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(
      MPI_Ireduce(msg.data(), nullptr, msg.count(), msg.mpi_type(), reduce_op, rank_root, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(reduceSend);
}

/// Given a GPU tile, perform MPI_Reduce in-place
template <class CommSender, class TileSender>
[[nodiscard]] auto scheduleReduceRecvInPlace(CommSender&& pcomm, MPI_Op reduce_op, TileSender&& tile) {
  // Note:
  //
  // GPU --> Duplicate --> (cCPU --> MPI --> cCPU) --> copy --> GPU
  //
  // where: cCPU = contiguous CPU
  using dlaf::comm::internal::reduceRecvInPlace_o;
  using dlaf::comm::internal::transformMPI;
  using dlaf::internal::CopyFromDestination;
  using dlaf::internal::CopyToDestination;
  using dlaf::internal::RequireContiguous;
  using dlaf::internal::whenAllLift;
  using dlaf::internal::withTemporaryTile;

  auto reduce_recv_in_place = [reduce_op,
                               pcomm = std::forward<CommSender>(pcomm)](auto const& tile_comm) mutable {
    return whenAllLift(std::move(pcomm), reduce_op, std::cref(tile_comm)) |
           transformMPI(reduceRecvInPlace_o);
  };

  // TODO: Can reductions happen on CPU only?
  return withTemporaryTile<Device::CPU, CopyToDestination::Yes, CopyFromDestination::Yes,
                           RequireContiguous::Yes>(std::forward<TileSender>(tile),
                                                   std::move(reduce_recv_in_place));
}

// TODO scheduleReduceSend with future will require to move the actual value, not the cref
/// Given a GPU tile perform MPI_Reduce in-place
template <class CommSender, class TileSender>
[[nodiscard]] auto scheduleReduceSend(CommSender&& pcomm, comm::IndexT_MPI rank_root, MPI_Op reduce_op,
                                      TileSender&& tile) {
  // Note:
  //
  // GPU --> Duplicate --> (cCPU --> MPI --> cCPU)
  //
  // where: cCPU = contiguous CPU
  using dlaf::comm::internal::reduceSend_o;
  using dlaf::comm::internal::transformMPI;
  using dlaf::internal::CopyFromDestination;
  using dlaf::internal::CopyToDestination;
  using dlaf::internal::RequireContiguous;
  using dlaf::internal::whenAllLift;
  using dlaf::internal::withTemporaryTile;

  auto reduce_send = [rank_root, reduce_op,
                      pcomm = std::forward<CommSender>(pcomm)](auto const& tile_comm) mutable {
    return whenAllLift(std::move(pcomm), rank_root, reduce_op, std::cref(tile_comm)) |
           transformMPI(reduceSend_o);
  };

  // TODO: Can reductions happen on CPU only?
  return withTemporaryTile<Device::CPU, CopyToDestination::Yes, CopyFromDestination::No,
                           RequireContiguous::Yes>(std::forward<TileSender>(tile),
                                                   std::move(reduce_send));
}
}
