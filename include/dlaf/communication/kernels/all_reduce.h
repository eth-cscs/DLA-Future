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

namespace dlaf {
namespace comm {

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
void scheduleAllReduce(CommSender&& pcomm, MPI_Op reduce_op,
                       pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_in,
                       pika::future<matrix::Tile<T, Device::CPU>> tile_out) {
  namespace ex = pika::execution::experimental;

  using pika::unwrapping;
  using pika::threads::thread_priority;

  using common::internal::ContiguousBufferHolder;
  using common::internal::makeItContiguous;
  using dlaf::internal::getBackendScheduler;
  using dlaf::internal::Policy;
  using dlaf::internal::transform;
  using dlaf::internal::whenAllLift;

  // Note:
  //
  //         +--------------------------------------+
  //         |                                      |
  // TILE_I -+-> makeContiguous -----> CONT_BUF_I --+--> mpi_call --> CONT_BUF_O --+
  //                                                |                              |
  // TILE_O ---> makeContiguous --+--> CONT_BUF_O --+                              |
  //                              |                                                |
  //                              +----------------------> TILE_O -----------------+-> copyBack

  return std::move(tile_out) |
         with_contiguous_comm_tile(
             [pcomm = std::forward<CommSender>(pcomm), reduce_op,
              tile_in =
                  std::move(tile_in)](const auto& tile_out,
                                      const auto& tile_out_contig_comm) -> ex::unique_any_sender<> {
               auto s = std::move(tile_in) |
                        with_contiguous_comm_tile([&, pcomm = std::forward<CommSender(pcomm)>,
                                                   reduce_op](const auto& tile_in,
                                                              const auto& tile_in_contig_comm) {
                          auto all_reduce_sender =
                              whenAllLift(std::move(pcomm), reduce_op, std::cref(tile_in_contig_comm),
                                          std::cref(tile_out_contig_comm)) |
                              transformMPI(internal::allReduce_o);
                        });

               // This is "copy back if needed".  Separate helper? copyIfNeeded?
               // operator== for Tile (the below is not 100% accurate if we have
               // views)? This should possibly be an option in
               // with_contiguous_comm_tile?
               if (tile_out.ptr() == tile_out_contig_comm.ptr()) {
                 return make_unique_any_sender(std::move(s));
               }
               else {
                 constexpr Device out_device_type = std::decay_t<decltype(tile_out)>::D;
                 constexpr Device comm_device_type = std::decay_t<decltype(tile_out_contig_comm)>::D;
                 constexpr Backend copy_backend =
                     dlaf::matrix::internal::CopyBackend_v<out_device_type, comm_device_type>;
                 // Copy the received data from the comm tile to the output tile.
                 return make_unique_any_sender(
                     std::move(s) | whenAllLift(std::cref(tile_out_contig_comm), std::cref(tile_out)) |
                     copy(Policy<copy_backend>(thread_priority::high)));
               }
             });
}

template <class CommSender, class TSender>
[[nodiscard]] auto scheduleAllReduceInPlace(CommSender&& pcomm, MPI_Op reduce_op, TSender&& tile) {
  namespace ex = pika::execution::experimental;

  using common::internal::makeItContiguous;
  using dlaf::comm::internal::make_unique_any_sender;
  using dlaf::comm::internal::with_contiguous_comm_tile;
  using dlaf::internal::getBackendScheduler;
  using dlaf::internal::keepIfSharedFuture;
  using dlaf::internal::Policy;
  using dlaf::internal::transform;
  using dlaf::internal::whenAllLift;
  using dlaf::matrix::copy;
  using pika::threads::thread_priority;

  // Note:
  //
  // TILE ---> makeContiguous --+--> CONT_BUF ----> mpi_call ---> CONT_BUF --+
  //                            |                                            |
  //                            +------------------> TILE -------------------+-> copyBack ---> TILE
  //
  //
  // The last TILE after the copyBack is returned so that other task can be attached to it,
  // AFTER the asynchronous MPI_AllReduce has completed
  return with_contiguous_comm_tile(std::forward<TSender>(tile),
                                   [reduce_op, pcomm = std::forward<CommSender>(
                                                   pcomm)](auto& tile_in,
                                                           auto const& tile_contig_comm) mutable {
                                     auto all_reduce_sender = whenAllLift(std::move(pcomm), reduce_op,
                                                                          std::cref(tile_contig_comm)) |
                                                              transformMPI(internal::allReduceInPlace_o);

                                     // This is "copy back if needed".  Separate
                                     // helper? copyIfNeeded?  operator== for
                                     // Tile (the below is not 100% accurate if
                                     // we have views)? This should possibly be
                                     // an option in with_contiguous_comm_tile?
                                     if (tile_in.ptr() == tile_contig_comm.ptr()) {
                                       return make_unique_any_sender(
                                           std::move(all_reduce_sender) |
                                           ex::then([&tile_in]() { return std::move(tile_in); }));
                                     }
                                     else {
                                       constexpr Device in_device_type =
                                           std::decay_t<decltype(tile_in)>::D;
                                       constexpr Device comm_device_type =
                                           std::decay_t<decltype(tile_contig_comm)>::D;
                                       constexpr Backend copy_backend =
                                           dlaf::matrix::internal::CopyBackend_v<in_device_type,
                                                                                 comm_device_type>;
                                       // Copy the received data from the
                                       // comm tile to the input tile.
                                       return make_unique_any_sender(
                                           whenAllLift(std::move(all_reduce_sender),
                                                       std::cref(tile_contig_comm), std::cref(tile_in)) |
                                           copy(Policy<copy_backend>(thread_priority::high)) |
                                           ex::then([&tile_in]() { return std::move(tile_in); }));
                                     }
                                   });
}
}
}
