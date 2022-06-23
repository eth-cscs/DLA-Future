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
#include "dlaf/matrix/tile.h"
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

template <class T>
auto allReduceInPlace(common::PromiseGuard<comm::Communicator> pcomm, MPI_Op reduce_op,
                      common::internal::ContiguousBufferHolder<T>& cont_buf, MPI_Request* req) {
  auto& comm = pcomm.ref();
  auto msg = comm::make_message(cont_buf.descriptor);

  DLAF_MPI_CHECK_ERROR(
      MPI_Iallreduce(MPI_IN_PLACE, msg.data(), msg.count(), msg.mpi_type(), reduce_op, comm, req));

  return std::move(cont_buf);
}

DLAF_MAKE_CALLABLE_OBJECT(allReduceInPlace);
}

template <class CommSender, class T>
void scheduleAllReduce(CommSender&& pcomm, MPI_Op reduce_op,
                       pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_in,
                       pika::future<matrix::Tile<T, Device::CPU>> tile_out) {
  namespace ex = pika::execution::experimental;

  using pika::unwrapping;
  using pika::execution::thread_priority;

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
  using dlaf::internal::getBackendScheduler;
  using dlaf::internal::Policy;
  using dlaf::internal::transform;
  using dlaf::internal::whenAllLift;
  using pika::execution::thread_priority;

  using T = dlaf::internal::SenderElementType<TSender>;

  // Note:
  //
  // TILE ---> makeContiguous --+--> CONT_BUF ----> mpi_call ---> CONT_BUF --+
  //                            |                                            |
  //                            +------------------> TILE -------------------+-> copyBack ---> TILE
  //
  //
  // The last TILE after the copyBack is returned so that other task can be attached to it,
  // AFTER the asynchronous MPI_AllReduce has completed
  return std::forward<TSender>(tile) | ex::transfer(getBackendScheduler<Backend::MC>()) |
         ex::let_value([pcomm = std::forward<CommSender>(pcomm),
                        reduce_op](matrix::Tile<T, Device::CPU>& tile) mutable {
           auto tile_reduced = whenAllLift(std::move(pcomm), reduce_op, makeItContiguous(tile)) |
                               transformMPI(internal::allReduceInPlace_o);
           return whenAllLift(std::move(tile_reduced), std::cref(tile)) |
                  transform(Policy<Backend::MC>(thread_priority::high), copyBack_o) |
                  ex::then([&tile]() { return std::move(tile); });
         });
}
}
}
