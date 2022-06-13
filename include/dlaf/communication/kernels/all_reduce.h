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

template <class T>
auto allReduce(common::PromiseGuard<comm::Communicator> pcomm, MPI_Op reduce_op,
               common::internal::ContiguousBufferHolder<const T>& cont_buf_in,
               common::internal::ContiguousBufferHolder<T>& cont_buf_out, MPI_Request* req) {
  auto& comm = pcomm.ref();
  auto msg_in = comm::make_message(cont_buf_in.descriptor);
  auto msg_out = comm::make_message(cont_buf_out.descriptor);

  DLAF_MPI_CHECK_ERROR(MPI_Iallreduce(msg_in.data(), msg_out.data(), msg_in.count(), msg_in.mpi_type(),
                                      reduce_op, comm, req));

  return std::move(cont_buf_out);
}

DLAF_MAKE_CALLABLE_OBJECT(allReduce);

template <class T, Device D>
auto allReduceInPlace(common::PromiseGuard<comm::Communicator> pcomm, MPI_Op reduce_op,
                      const matrix::Tile<T, D>& tile, MPI_Request* req) {
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
  using common::internal::copyBack_o;
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
  auto f = unwrapping([pcomm = std::forward<CommSender>(pcomm),
                       reduce_op](const matrix::Tile<const T, Device::CPU>& tile_in,
                                  matrix::Tile<T, Device::CPU>& tile_out) mutable {
    auto tile_reduced =
        whenAllLift(std::move(pcomm), reduce_op, makeItContiguous(tile_in), makeItContiguous(tile_out)) |
        transformMPI(internal::allReduce_o);
    return whenAllLift(std::move(tile_reduced), std::cref(tile_out)) |
           transform(Policy<Backend::MC>(thread_priority::high), copyBack_o);
  });
  ex::when_all(ex::keep_future(std::move(tile_in)), std::move(tile_out)) |
      ex::transfer(getBackendScheduler<Backend::MC>()) | ex::let_value(std::move(f)) |
      ex::start_detached();
}

template <class CommSender, class TSender>
[[nodiscard]] auto scheduleAllReduceInPlace(CommSender&& pcomm, MPI_Op reduce_op, TSender&& tile) {
  namespace ex = pika::execution::experimental;

  using common::internal::copyBack_o;
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
