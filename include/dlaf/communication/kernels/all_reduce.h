//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <mpi.h>

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
auto allReduce(common::PromiseGuard<comm::Communicator> pcomm, MPI_Op reduce_op,
               common::internal::ContiguousBufferHolder<const T> bag_in,
               common::internal::ContiguousBufferHolder<T> bag_out,
               matrix::Tile<const T, Device::CPU> const&, MPI_Request* req) {
  auto& comm = pcomm.ref();
  auto msg_in = comm::make_message(bag_in.descriptor);
  auto msg_out = comm::make_message(bag_out.descriptor);

  DLAF_MPI_CALL(MPI_Iallreduce(msg_in.data(), msg_out.data(), msg_in.count(), msg_in.mpi_type(),
                               reduce_op, comm, req));

  return bag_out;
}

DLAF_MAKE_CALLABLE_OBJECT(allReduce);

template <class T>
auto allReduceInPlace(common::PromiseGuard<comm::Communicator> pcomm, MPI_Op reduce_op,
                      common::internal::ContiguousBufferHolder<T> bag, MPI_Request* req) {
  auto& comm = pcomm.ref();
  auto msg = comm::make_message(bag.descriptor);

  DLAF_MPI_CALL(
      MPI_Iallreduce(MPI_IN_PLACE, msg.data(), msg.count(), msg.mpi_type(), reduce_op, comm, req));

  return bag;
}

DLAF_MAKE_CALLABLE_OBJECT(allReduceInPlace);
}

template <class T>
void scheduleAllReduce(const comm::Executor& ex,
                       hpx::future<common::PromiseGuard<comm::Communicator>> pcomm, MPI_Op reduce_op,
                       hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_in,
                       hpx::future<matrix::Tile<T, Device::CPU>> tile_out) {
  using hpx::dataflow;
  using hpx::util::unwrapping;

  using common::internal::ContiguousBufferHolder;
  using common::internal::copyBack_o;
  using common::internal::makeItContiguous_o;
  using matrix::unwrapExtendTiles;

  // Note:
  //
  //         +---------------------------------+
  //         |                                 |
  // TILE_I -+-> makeContiguous -----> BAG_I --+--> mpi_call --> BAG_O --+
  //                                           |                         |
  // TILE_O ---> makeContiguous --+--> BAG_O --+                         |
  //                              |                                      |
  //                              +----------------> TILE_O -------------+-> copyBack

  auto ex_copy = getHpExecutor<Backend::MC>();

  hpx::future<ContiguousBufferHolder<const T>> bag_in =
      dataflow(unwrapping(makeItContiguous_o), tile_in);

  hpx::future<ContiguousBufferHolder<T>> bag_out;
  {
    auto wrapped = getUnwrapRetValAndArgs(
        dataflow(ex_copy, unwrapExtendTiles(makeItContiguous_o), std::move(tile_out)));
    bag_out = std::move(wrapped.first);
    tile_out = std::move(hpx::get<0>(wrapped.second));
  }

  bag_out = getUnwrapReturnValue(dataflow(ex, unwrapExtendTiles(internal::allReduce_o), std::move(pcomm),
                                          reduce_op, std::move(bag_in), std::move(bag_out), tile_in));

  hpx::dataflow(ex_copy, unwrapping(copyBack_o), std::move(tile_out), std::move(bag_out));
}

template <class T>
hpx::future<matrix::Tile<T, Device::CPU>> scheduleAllReduceInPlace(
    const comm::Executor& ex, hpx::future<common::PromiseGuard<comm::Communicator>> pcomm,
    MPI_Op reduce_op, hpx::future<matrix::Tile<T, Device::CPU>> tile) {
  using hpx::dataflow;
  using hpx::util::unwrapping;

  using common::internal::copyBack_o;
  using common::internal::makeItContiguous_o;
  using matrix::unwrapExtendTiles;

  // Note:
  //
  // TILE ---> makeContiguous --+--> BAG ----> mpi_call ---> BAG --+
  //                            |                                  |
  //                            +-------------> TILE --------------+-> copyBack ---> TILE
  //
  //
  // The last TILE after the copyBack is returned so that other task can be attached to it,
  // AFTER the asynchronous MPI_AllReduce has completed

  auto ex_copy = getHpExecutor<Backend::MC>();

  hpx::future<common::internal::ContiguousBufferHolder<T>> bag;
  {
    auto wrapped = getUnwrapRetValAndArgs(
        dataflow(ex_copy, unwrapExtendTiles(makeItContiguous_o), std::move(tile)));
    bag = std::move(wrapped.first);
    tile = std::move(hpx::get<0>(wrapped.second));
  }

  bag = dataflow(ex, unwrapping(internal::allReduceInPlace_o), std::move(pcomm), reduce_op,
                 std::move(bag));

  return dataflow(ex_copy, unwrapping(copyBack_o), std::move(tile), std::move(bag));
}
}
}
