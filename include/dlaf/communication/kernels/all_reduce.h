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
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"
#include "dlaf/communication/rdma.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/transform_mpi.h"
#include "dlaf/sender/with_temporary_tile.h"

namespace dlaf::comm {
namespace internal {
template <class T, Device D>
auto allReduce(const Communicator& comm, MPI_Op reduce_op, const matrix::Tile<const T, D>& tile_in,
               const matrix::Tile<T, D>& tile_out, MPI_Request* req) {
  static_assert(D == Device::CPU, "allReduce requires CPU memory");
  DLAF_ASSERT(tile_in.is_contiguous(), "");
  DLAF_ASSERT(tile_out.is_contiguous(), "");

  auto msg_in = comm::make_message(common::make_data(tile_in));
  auto msg_out = comm::make_message(common::make_data(tile_out));
  DLAF_MPI_CHECK_ERROR(MPI_Iallreduce(msg_in.data(), msg_out.data(), msg_in.count(), msg_in.mpi_type(),
                                      reduce_op, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(allReduce);

template <class T, Device D>
auto allReduceInPlace(const Communicator& comm, MPI_Op reduce_op, const matrix::Tile<T, D>& tile,
                      MPI_Request* req) {
  static_assert(D == Device::CPU, "allReduceInPlace requires CPU memory");
  DLAF_ASSERT(tile.is_contiguous(), "");

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(
      MPI_Iallreduce(MPI_IN_PLACE, msg.data(), msg.count(), msg.mpi_type(), reduce_op, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(allReduceInPlace);
}

/// Schedule an all reduce.
///
/// An input and output tile is required for the reduction. The returned sender
/// signals completion when the reduction is done. The output sender tile must
/// be writable so that the received and reduced data can be written to it. The
/// output tile is sent by the returned sender.
template <class CommSender, class TileInSender, class TileOutSender>
[[nodiscard]] auto scheduleAllReduce(CommSender&& pcomm, MPI_Op reduce_op, TileInSender&& tile_in,
                                     TileOutSender&& tile_out) {
  using dlaf::comm::internal::allReduce_o;
  using dlaf::comm::internal::transformMPI;
  using dlaf::internal::CopyFromDestination;
  using dlaf::internal::CopyToDestination;
  using dlaf::internal::RequireContiguous;
  using dlaf::internal::whenAllLift;
  using dlaf::internal::withTemporaryTile;

  // We create two nested scopes for the input and output tiles with
  // withTemporaryTile. The output tile is in the outer scope as the output tile
  // will be returned by the returned sender.
  auto all_reduce_final = [reduce_op, pcomm = std::forward<CommSender>(pcomm),
                           tile_in =
                               std::forward<TileInSender>(tile_in)](auto const& tile_out_comm) mutable {
    auto all_reduce = [reduce_op, pcomm = std::move(pcomm),
                       &tile_out_comm](auto const& tile_in_comm) mutable {
      return whenAllLift(std::move(pcomm), reduce_op, std::cref(tile_in_comm),
                         std::cref(tile_out_comm)) |
             transformMPI(allReduce_o);
    };
    // The input tile must be copied to the temporary tile used for the
    // reduction, but the temporary tile does not need to be copied back to the
    // input since the data is not changed by the reduction (the result is
    // written into the output tile instead).  The reduction is explicitly done
    // on CPU memory so that we can manage potential asynchronous copies between
    // CPU and GPU. A reduction requires contiguous memory.
    return withTemporaryTile<Device::CPU, CopyToDestination::Yes, CopyFromDestination::No,
                             RequireContiguous::Yes>(std::move(tile_in), std::move(all_reduce));
  };

  // The output tile does not need to be copied to the temporary tile since it
  // is only written to. The written data is copied back from the temporary tile
  // to the output tile. The reduction is explicitly done on CPU memory so that
  // we can manage potential asynchronous copies between CPU and GPU. A
  // reduction requires contiguous memory.
  return withTemporaryTile<Device::CPU, CopyToDestination::No, CopyFromDestination::Yes,
                           RequireContiguous::Yes>(std::forward<TileOutSender>(tile_out),
                                                   std::move(all_reduce_final));
}

/// Schedule an in-place all reduce.
///
/// The returned sender signals completion when the reduction is done.  The
/// sender tile must be writable so that the received and reduced data can be
/// written to it. The tile is sent by the returned sender.
template <class CommSender, class TileSender>
[[nodiscard]] auto scheduleAllReduceInPlace(CommSender&& pcomm, MPI_Op reduce_op, TileSender&& tile) {
  using dlaf::comm::internal::allReduceInPlace_o;
  using dlaf::comm::internal::transformMPI;
  using dlaf::internal::CopyFromDestination;
  using dlaf::internal::CopyToDestination;
  using dlaf::internal::RequireContiguous;
  using dlaf::internal::whenAllLift;
  using dlaf::internal::withTemporaryTile;

  auto all_reduce_in_place = [reduce_op,
                              pcomm = std::forward<CommSender>(pcomm)](auto const& tile_comm) mutable {
    return whenAllLift(std::move(pcomm), reduce_op, std::cref(tile_comm)) |
           transformMPI(allReduceInPlace_o);
  };

  // The tile has to be copied both to and from the temporary tile since the
  // reduction is done in-place. The reduction is explicitly done on CPU memory
  // so that we can manage potential asynchronous copies between CPU and GPU. A
  // reduction requires contiguous memory.
  return withTemporaryTile<Device::CPU, CopyToDestination::Yes, CopyFromDestination::Yes,
                           RequireContiguous::Yes>(std::forward<TileSender>(tile),
                                                   std::move(all_reduce_in_place));
}
}
