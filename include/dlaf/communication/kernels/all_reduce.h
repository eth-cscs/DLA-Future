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
#include <pika/mpi.hpp>
#include <pika/unwrap.hpp>

#include "dlaf/common/callable_object.h"
#include "dlaf/common/contiguous_buffer_holder.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"
#include "dlaf/communication/rdma.h"
#include "dlaf/matrix/tile.h"

namespace dlaf {
namespace comm {

namespace internal {

template <class T>
auto allReduce(const common::PromiseGuard<comm::Communicator>& pcomm, MPI_Op reduce_op,
               common::internal::ContiguousBufferHolder<const T> cont_buf_in,
               common::internal::ContiguousBufferHolder<T> cont_buf_out, MPI_Request* req) {
  auto& comm = pcomm.ref();
  auto msg_in = comm::make_message(cont_buf_in.descriptor);
  auto msg_out = comm::make_message(cont_buf_out.descriptor);

  DLAF_MPI_CHECK_ERROR(MPI_Iallreduce(msg_in.data(), msg_out.data(), msg_in.count(), msg_in.mpi_type(),
                                      reduce_op, comm, req));
  return cont_buf_out;
}

DLAF_MAKE_CALLABLE_OBJECT(allReduce);

template <class T>
auto allReduceInPlace(const common::PromiseGuard<comm::Communicator>& pcomm, MPI_Op reduce_op,
                      common::internal::ContiguousBufferHolder<T> cont_buf, MPI_Request* req) {
  auto& comm = pcomm.ref();
  auto msg = comm::make_message(cont_buf.descriptor);

  DLAF_MPI_CHECK_ERROR(
      MPI_Iallreduce(MPI_IN_PLACE, msg.data(), msg.count(), msg.mpi_type(), reduce_op, comm, req));
  return cont_buf;
}

DLAF_MAKE_CALLABLE_OBJECT(allReduceInPlace);
}

template <class T>
void scheduleAllReduce(pika::future<common::PromiseGuard<comm::Communicator>> pcomm, MPI_Op reduce_op,
                       pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_in,
                       pika::future<matrix::Tile<T, Device::CPU>> tile_out) {
  using pika::execution::experimental::keep_future;
  using pika::execution::experimental::let_value;
  using pika::execution::experimental::start_detached;
  using pika::execution::experimental::transfer;
  using pika::execution::experimental::when_all;
  using pika::mpi::experimental::transform_mpi;
  using pika::threads::thread_priority;
  using pika::unwrapping;

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
  when_all(keep_future(std::move(tile_in)), std::move(tile_out)) |
      transfer(getBackendScheduler<Backend::MC>()) |
      let_value(unwrapping(
          [pcomm = std::move(pcomm), reduce_op](const matrix::Tile<const T, Device::CPU>& tile_in,
                                                matrix::Tile<T, Device::CPU>& tile_out) mutable {
            return whenAllLift(std::move(pcomm), reduce_op, makeItContiguous(tile_in),
                               makeItContiguous(tile_out)) |
                   transform_mpi(unwrapping(internal::allReduce_o)) |
                   transform(Policy<Backend::MC>(thread_priority::high),
                             std::bind(copyBack_o, std::placeholders::_1, std::cref(tile_out)));
          })) |
      start_detached();
}

template <class T>
pika::future<matrix::Tile<T, Device::CPU>> scheduleAllReduceInPlace(
    pika::future<common::PromiseGuard<comm::Communicator>> pcomm, MPI_Op reduce_op,
    pika::future<matrix::Tile<T, Device::CPU>> tile) {
  using pika::execution::experimental::let_value;
  using pika::execution::experimental::make_future;
  using pika::execution::experimental::then;
  using pika::execution::experimental::transfer;
  using pika::mpi::experimental::transform_mpi;
  using pika::unwrapping;

  using common::internal::copyBack_o;
  using common::internal::makeItContiguous;
  using dlaf::internal::getBackendScheduler;
  using dlaf::internal::Policy;
  using dlaf::internal::transform;
  using dlaf::internal::whenAllLift;

  // Note:
  //
  // TILE ---> makeContiguous --+--> CONT_BUF ----> mpi_call ---> CONT_BUF --+
  //                            |                                            |
  //                            +------------------> TILE -------------------+-> copyBack ---> TILE
  //
  //
  // The last TILE after the copyBack is returned so that other task can be attached to it,
  // AFTER the asynchronous MPI_AllReduce has completed
  return std::move(tile) | transfer(getBackendScheduler<Backend::MC>()) |
         let_value([pcomm = std::move(pcomm), reduce_op](matrix::Tile<T, Device::CPU>& tile) mutable {
           return whenAllLift(std::move(pcomm), reduce_op, makeItContiguous(tile)) |
                  transform_mpi(unwrapping(internal::allReduceInPlace_o)) |
                  transform(Policy<Backend::MC>(),
                            std::bind(copyBack_o, std::placeholders::_1, std::cref(tile))) |
                  then([&tile]() { return std::move(tile); });
         }) |
         make_future();
}
}
}
