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

/// @file

#include <mpi.h>

#include <pika/execution.hpp>
#include <pika/unwrap.hpp>

#include "dlaf/common/callable_object.h"
#include "dlaf/common/contiguous_buffer_holder.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"
#include "dlaf/communication/rdma.h"
#include "dlaf/communication/with_communication_tile.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/transform_mpi.h"

namespace dlaf::comm {
namespace internal {
template <class T, Device D>
auto allReduce(common::PromiseGuard<comm::Communicator> pcomm, MPI_Op reduce_op,
               const matrix::Tile<const T, D>& tile_in, const matrix::Tile<T, D>& tile_out,
               MPI_Request* req) {
  // TODO: Assert CPU tile?
  DLAF_ASSERT(tile_in.is_contiguous(), "");
  DLAF_ASSERT(tile_out.is_contiguous(), "");

  auto& comm = pcomm.ref();
  auto msg_in = comm::make_message(common::make_data(tile_in));
  auto msg_out = comm::make_message(common::make_data(tile_out));
  DLAF_MPI_CHECK_ERROR(MPI_Iallreduce(msg_in.data(), msg_out.data(), msg_in.count(), msg_in.mpi_type(),
                                      reduce_op, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(allReduce);

template <class T, Device D>
auto allReduceInPlace(common::PromiseGuard<comm::Communicator> pcomm, MPI_Op reduce_op,
                      const matrix::Tile<T, D>& tile, MPI_Request* req) {
  // TODO: Assert CPU tile?
  DLAF_ASSERT(tile.is_contiguous(), "");

  auto& comm = pcomm.ref();
  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(
      MPI_Iallreduce(MPI_IN_PLACE, msg.data(), msg.count(), msg.mpi_type(), reduce_op, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(allReduceInPlace);
}

template <class CommSender, class TileInSender, class TileOutSender>
[[nodiscard]] auto scheduleAllReduce(CommSender&& pcomm, MPI_Op reduce_op, TileInSender&& tile_in,
                                     TileOutSender&& tile_out) {
  // Note:
  //
  //         +--------------------------------------+
  //         |                                      |
  // TILE_I -+-> makeContiguous -----> CONT_BUF_I --+--> mpi_call --> CONT_BUF_O --+
  //                                                |                              |
  // TILE_O ---> makeContiguous --+--> CONT_BUF_O --+                              |
  //                              |                                                |
  //                              +----------------------> TILE_O -----------------+-> copyBack
  using dlaf::comm::internal::allReduce_o;
  using dlaf::comm::internal::CopyFromDestination;
  using dlaf::comm::internal::CopyToDestination;
  using dlaf::comm::internal::RequireContiguous;
  using dlaf::comm::internal::transformMPI;
  using dlaf::comm::internal::withTemporaryTile;
  using dlaf::internal::whenAllLift;

  auto all_reduce_final = [reduce_op, pcomm = std::forward<CommSender>(pcomm),
                           tile_in =
                               std::forward<TileInSender>(tile_in)](auto const& tile_out_comm) mutable {
    auto all_reduce = [reduce_op, pcomm = std::move(pcomm),
                       &tile_out_comm](auto const& tile_in_comm) mutable {
      return whenAllLift(std::move(pcomm), reduce_op, std::cref(tile_in_comm),
                         std::cref(tile_out_comm)) |
             transformMPI(allReduce_o);
    };
    return withTemporaryTile<Device::CPU, CopyToDestination::Yes, CopyFromDestination::No,
                             RequireContiguous::Yes>(std::move(tile_in), std::move(all_reduce));
  };

  // TODO: Can reductions happen on CPU only?
  return withTemporaryTile<Device::CPU, CopyToDestination::No, CopyFromDestination::Yes,
                           RequireContiguous::Yes>(std::forward<TileOutSender>(tile_out),
                                                   std::move(all_reduce_final));
}

template <class CommSender, class TileSender>
[[nodiscard]] auto scheduleAllReduceInPlace(CommSender&& pcomm, MPI_Op reduce_op, TileSender&& tile) {
  // Note:
  //
  // TILE ---> makeContiguous --+--> CONT_BUF ----> mpi_call ---> CONT_BUF --+
  //                            |                                            |
  //                            +------------------> TILE -------------------+-> copyBack ---> TILE
  //
  // The last TILE after the copyBack is returned so that other task can be attached to it,
  // AFTER the asynchronous MPI_AllReduce has completed
  using dlaf::comm::internal::allReduceInPlace_o;
  using dlaf::comm::internal::CopyFromDestination;
  using dlaf::comm::internal::CopyToDestination;
  using dlaf::comm::internal::RequireContiguous;
  using dlaf::comm::internal::transformMPI;
  using dlaf::comm::internal::withTemporaryTile;
  using dlaf::internal::whenAllLift;

  auto all_reduce_in_place = [reduce_op,
                              pcomm = std::forward<CommSender>(pcomm)](auto const& tile_comm) mutable {
    return whenAllLift(std::move(pcomm), reduce_op, std::cref(tile_comm)) |
           transformMPI(allReduceInPlace_o);
  };

  // TODO: Can reductions happen on CPU only?
  return withTemporaryTile<Device::CPU, CopyToDestination::Yes, CopyFromDestination::Yes,
                           RequireContiguous::Yes>(std::forward<TileSender>(tile),
                                                   std::move(all_reduce_in_place));
}
}
