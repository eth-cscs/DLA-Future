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
#include <pika/future.hpp>
#include <pika/unwrap.hpp>

#include "dlaf/common/callable_object.h"
#include "dlaf/common/contiguous_buffer_holder.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"
#include "dlaf/communication/rdma.h"
#include "dlaf/communication/with_contiguous_buffer.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/keep_if_shared_future.h"
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

template <class CommSender, class T>
[[nodiscard]] auto scheduleAllReduce(CommSender&& pcomm, MPI_Op reduce_op,
                                     pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_in,
                                     pika::future<matrix::Tile<T, Device::CPU>> tile_out) {
  namespace ex = pika::execution::experimental;

  using dlaf::comm::internal::copyBack;
  using dlaf::comm::internal::withContiguousCommTile;
  using dlaf::internal::whenAllLift;
  using pika::unwrapping;

  // Note:
  //
  //         +--------------------------------------+
  //         |                                      |
  // TILE_I -+-> makeContiguous -----> CONT_BUF_I --+--> mpi_call --> CONT_BUF_O --+
  //                                                |                              |
  // TILE_O ---> makeContiguous --+--> CONT_BUF_O --+                              |
  //                              |                                                |
  //                              +----------------------> TILE_O -----------------+-> copyBack
  auto all_reduce_copy_back = [pcomm = std::forward<CommSender>(pcomm), reduce_op,
                               tile_in = std::move(tile_in)](const auto& tile_out,
                                                             const auto& tile_out_contig_comm) mutable {
    auto all_reduce = unwrapping([&tile_out_contig_comm, pcomm = std::move(pcomm),
                                  reduce_op](const auto&, const auto& tile_in_contig_comm) mutable {
      return whenAllLift(std::move(pcomm), reduce_op, std::cref(tile_in_contig_comm),
                         std::cref(tile_out_contig_comm)) |
             transformMPI(internal::allReduce_o);
    });
    auto all_reduce_sender =
        withContiguousCommTile(ex::keep_future(std::move(tile_in)), std::move(all_reduce));
    return copyBack(std::move(all_reduce_sender), tile_out, tile_out_contig_comm);
  };
  return withContiguousCommTile(std::move(tile_out), std::move(all_reduce_copy_back));
}

template <class CommSender, class TSender>
[[nodiscard]] auto scheduleAllReduceInPlace(CommSender&& pcomm, MPI_Op reduce_op, TSender&& tile) {
  namespace ex = pika::execution::experimental;

  using dlaf::comm::internal::copyBack;
  using dlaf::comm::internal::withContiguousCommTile;
  using dlaf::internal::whenAllLift;

  // Note:
  //
  // TILE ---> makeContiguous --+--> CONT_BUF ----> mpi_call ---> CONT_BUF --+
  //                            |                                            |
  //                            +------------------> TILE -------------------+-> copyBack ---> TILE
  //
  // The last TILE after the copyBack is returned so that other task can be attached to it,
  // AFTER the asynchronous MPI_AllReduce has completed
  auto all_reduce_in_place_copy_back =
      [reduce_op, pcomm = std::forward<CommSender>(pcomm)](auto& tile_in,
                                                           auto const& tile_contig_comm) mutable {
        auto all_reduce_sender = whenAllLift(std::move(pcomm), reduce_op, std::cref(tile_contig_comm)) |
                                 transformMPI(internal::allReduceInPlace_o);
        return copyBack(std::move(all_reduce_sender), tile_in, tile_contig_comm) |
               ex::then([&tile_in]() { return std::move(tile_in); });
      };
  return withContiguousCommTile(std::forward<TSender>(tile), std::move(all_reduce_in_place_copy_back));
}
}
