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
#include "dlaf/communication/executor.h"
#include "dlaf/communication/message.h"
#include "dlaf/communication/rdma.h"
#include "dlaf/executors.h"
#include "dlaf/matrix/tile.h"

namespace dlaf {
namespace comm {

namespace internal {

template <class T>
auto allReduce(const common::PromiseGuard<comm::Communicator>& pcomm, MPI_Op reduce_op,
               common::internal::ContiguousBufferHolder<const T> cont_buf_in,
               common::internal::ContiguousBufferHolder<T> cont_buf_out,
               const matrix::Tile<const T, Device::CPU>&, MPI_Request* req) {
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
void scheduleAllReduce(const comm::Executor& ex,
                       pika::future<common::PromiseGuard<comm::Communicator>> pcomm, MPI_Op reduce_op,
                       pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_in,
                       pika::future<matrix::Tile<T, Device::CPU>> tile_out) {
  using pika::dataflow;
  using pika::execution::experimental::ensure_started;
  using pika::execution::experimental::keep_future;
  using pika::execution::experimental::make_future;
  using pika::execution::experimental::start_detached;
  using pika::mpi::experimental::transform_mpi;
  using pika::threads::thread_priority;
  using pika::unwrapping;

  using common::internal::ContiguousBufferHolder;
  using common::internal::copyBack_o;
  using common::internal::makeItContiguous_o;
  using dlaf::internal::Policy;
  using dlaf::internal::transform;
  using dlaf::internal::whenAllLift;
  using matrix::unwrapExtendTiles;

  // Note:
  //
  //         +--------------------------------------+
  //         |                                      |
  // TILE_I -+-> makeContiguous -----> CONT_BUF_I --+--> mpi_call --> CONT_BUF_O --+
  //                                                |                              |
  // TILE_O ---> makeContiguous --+--> CONT_BUF_O --+                              |
  //                              |                                                |
  //                              +----------------------> TILE_O -----------------+-> copyBack

  auto ex_copy = getHpExecutor<Backend::MC>();

  pika::future<ContiguousBufferHolder<const T>> cont_buf_in =
      keep_future(tile_in) | transform(Policy<Backend::MC>(), makeItContiguous_o) | make_future();

  pika::future<ContiguousBufferHolder<T>> cont_buf_out;
  {
    // TODO: Need something similar to split_future, but not quite. Can tile_out
    // be passed differently? Can we use a shared_future instead?
    auto wrapped = getUnwrapRetValAndArgs(
        dataflow(ex_copy, unwrapExtendTiles(makeItContiguous_o), std::move(tile_out)));
    cont_buf_out = std::move(wrapped.first);
    tile_out = std::move(pika::get<0>(wrapped.second));
  }

  cont_buf_out = whenAllLift(std::move(pcomm), reduce_op, std::move(cont_buf_in),
                             std::move(cont_buf_out), keep_future(std::move(tile_in))) |
                 transform_mpi(pika::unwrapping(internal::allReduce_o)) | make_future();

  when_all(std::move(cont_buf_out), std::move(tile_out)) |
      transform(Policy<Backend::MC>(thread_priority::high), copyBack_o) | start_detached();
}

template <class T>
pika::future<matrix::Tile<T, Device::CPU>> scheduleAllReduceInPlace(
    const comm::Executor& ex, pika::future<common::PromiseGuard<comm::Communicator>> pcomm,
    MPI_Op reduce_op, pika::future<matrix::Tile<T, Device::CPU>> tile) {
  using pika::dataflow;
  using pika::execution::experimental::make_future;
  using pika::mpi::experimental::transform_mpi;
  using pika::unwrapping;

  using common::internal::copyBack_o;
  using common::internal::makeItContiguous_o;
  using dlaf::internal::whenAllLift;
  using matrix::unwrapExtendTiles;

  // Note:
  //
  // TILE ---> makeContiguous --+--> CONT_BUF ----> mpi_call ---> CONT_BUF --+
  //                            |                                            |
  //                            +------------------> TILE -------------------+-> copyBack ---> TILE
  //
  //
  // The last TILE after the copyBack is returned so that other task can be attached to it,
  // AFTER the asynchronous MPI_AllReduce has completed

  auto ex_copy = getHpExecutor<Backend::MC>();

  pika::future<common::internal::ContiguousBufferHolder<T>> cont_buf;
  {
    auto wrapped = getUnwrapRetValAndArgs(
        dataflow(ex_copy, unwrapExtendTiles(makeItContiguous_o), std::move(tile)));
    cont_buf = std::move(wrapped.first);
    tile = std::move(pika::get<0>(wrapped.second));
  }

  cont_buf = whenAllLift(std::move(pcomm), reduce_op, std::move(cont_buf)) |
             transform_mpi(pika::unwrapping(internal::allReduceInPlace_o)) | make_future();

  // Note:
  // This extracts the tile given as argument to copyBack, not the return value.
  return pika::get<1>(pika::split_future(
      dataflow(ex_copy, matrix::unwrapExtendTiles(copyBack_o), std::move(cont_buf), std::move(tile))));
}
}
}
