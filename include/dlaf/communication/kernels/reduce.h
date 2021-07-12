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
#include "dlaf/common/data.h"
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
auto reduceRecvInPlace(common::PromiseGuard<comm::Communicator> pcomm, MPI_Op reduce_op,
                       common::internal::ContiguousBufferHolder<T> bag, MPI_Request* req) {
  auto msg = comm::make_message(bag.descriptor);
  auto& comm = pcomm.ref();

  DLAF_MPI_CALL(MPI_Ireduce(MPI_IN_PLACE, msg.data(), msg.count(), msg.mpi_type(), reduce_op,
                            comm.rank(), comm, req));

  return bag;
}

DLAF_MAKE_CALLABLE_OBJECT(reduceRecvInPlace);

template <class T>
auto reduceSend(comm::IndexT_MPI rank_root, common::PromiseGuard<comm::Communicator> pcomm,
                MPI_Op reduce_op, common::internal::ContiguousBufferHolder<const T> bag,
                matrix::Tile<const T, Device::CPU> const&, MPI_Request* req) {
  auto msg = comm::make_message(bag.descriptor);
  auto& comm = pcomm.ref();

  DLAF_MPI_CALL(
      MPI_Ireduce(msg.data(), nullptr, msg.count(), msg.mpi_type(), reduce_op, rank_root, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(reduceSend);

}

template <class T>
void scheduleReduceRecvInPlace(const comm::Executor& ex,
                               hpx::future<common::PromiseGuard<comm::Communicator>> pcomm,
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
  //                            +-------------> TILE --------------+-> copyBack

  auto ex_copy = getHpExecutor<Backend::MC>();

  hpx::future<common::internal::ContiguousBufferHolder<T>> bag;
  {
    auto wrapped = getUnwrapRetValAndArgs(
        dataflow(ex_copy, unwrapExtendTiles(makeItContiguous_o), std::move(tile)));

    bag = std::move(wrapped.first);
    tile = std::move(hpx::get<0>(wrapped.second));
  }

  bag = dataflow(ex, unwrapping(internal::reduceRecvInPlace_o), std::move(pcomm), reduce_op,
                 std::move(bag));

  dataflow(ex_copy, unwrapping(copyBack_o), std::move(tile), std::move(bag));
}

template <class T>
void scheduleReduceSend(const comm::Executor& ex, comm::IndexT_MPI rank_root,
                        hpx::future<common::PromiseGuard<comm::Communicator>> pcomm, MPI_Op reduce_op,
                        hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile) {
  using hpx::dataflow;
  using hpx::util::unwrapping;

  using common::internal::makeItContiguous_o;
  using matrix::unwrapExtendTiles;

  // Note:
  //
  // TILE -+-> makeContiguous --+--> BAG ---+--> mpi_call
  //       |                                |
  //       +--------------> TILE -----------+

  auto ex_copy = getHpExecutor<Backend::MC>();

  // TODO shared_future<Tile> as assumption, it requires changes for future<Tile>
  hpx::future<common::internal::ContiguousBufferHolder<const T>> bag =
      dataflow(ex_copy, unwrapping(makeItContiguous_o), tile);

  dataflow(ex, unwrapExtendTiles(internal::reduceSend_o), rank_root, std::move(pcomm), reduce_op,
           std::move(bag), tile);
}
}
}
