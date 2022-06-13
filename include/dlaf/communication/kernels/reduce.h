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
#include <pika/future.hpp>
#include <pika/unwrap.hpp>

#include "dlaf/common/contiguous_buffer_holder.h"
#include "dlaf/common/data.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"
#include "dlaf/communication/with_contiguous_buffer.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/schedulers.h"
#include "dlaf/sender/traits.h"
#include "dlaf/sender/transform_mpi.h"

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
void reduceSend(comm::IndexT_MPI rank_root, common::PromiseGuard<comm::Communicator> pcomm,
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
  using dlaf::comm::internal::copyBack;
  using dlaf::comm::internal::transformMPI;
  using dlaf::comm::internal::withSimilarContiguousCommTile;
  using dlaf::internal::whenAllLift;

  auto reduce_recv_in_place_copy_back =
      [reduce_op, pcomm = std::forward<CommSender>(pcomm)](auto const& tile_in,
                                                           auto const& tile_contig_comm) mutable {
        auto recv_sender = whenAllLift(std::move(pcomm), reduce_op, std::cref(tile_contig_comm)) |
                           transformMPI(internal::reduceRecvInPlace_o);
        return copyBack(std::move(recv_sender), tile_in, tile_contig_comm);
      };
  return withSimilarContiguousCommTile(std::forward<TileSender>(tile),
                                       std::move(reduce_recv_in_place_copy_back));
}

// TODO scheduleReduceSend with future will require to move the actual value, not the cref
/// Given a GPU tile perform MPI_Reduce in-place
template <class CommSender, class TileSender>
[[nodiscard]] auto scheduleReduceSend(comm::IndexT_MPI rank_root, CommSender&& pcomm, MPI_Op reduce_op,
                                      TileSender&& tile) {
  // Note:
  //
  // GPU --> Duplicate --> (cCPU --> MPI --> cCPU)
  //
  // where: cCPU = contiguous CPU
  namespace ex = pika::execution::experimental;

  using dlaf::comm::internal::transformMPI;
  using dlaf::comm::internal::withContiguousCommTile;
  using dlaf::internal::whenAllLift;

  auto reduce_send = [rank_root, reduce_op,
                      pcomm = std::forward<CommSender>(pcomm)](auto const&,
                                                               auto const& tile_contig_comm) mutable {
    return whenAllLift(rank_root, std::move(pcomm), reduce_op, std::cref(tile_contig_comm)) |
           transformMPI(internal::reduceSend_o);
  };
  return withContiguousCommTile(std::forward<TileSender>(tile), std::move(reduce_send));
}
}
