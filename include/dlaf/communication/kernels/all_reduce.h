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

#include <hpx/local/unwrap.hpp>

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
               common::internal::ContiguousBufferHolder<const T> cont_buf_in,
               common::internal::ContiguousBufferHolder<T> cont_buf_out,
               matrix::Tile<const T, Device::CPU> const&, MPI_Request* req) {
  auto& comm = pcomm.ref();
  auto msg_in = comm::make_message(cont_buf_in.descriptor);
  auto msg_out = comm::make_message(cont_buf_out.descriptor);

  DLAF_MPI_CALL(MPI_Iallreduce(msg_in.data(), msg_out.data(), msg_in.count(), msg_in.mpi_type(),
                               reduce_op, comm, req));
  return cont_buf_out;
}

DLAF_MAKE_CALLABLE_OBJECT(allReduce);

template <class T>
auto allReduceInPlace(common::PromiseGuard<comm::Communicator> pcomm, MPI_Op reduce_op,
                      common::internal::ContiguousBufferHolder<T> cont_buf, MPI_Request* req) {
  auto& comm = pcomm.ref();
  auto msg = comm::make_message(cont_buf.descriptor);

  DLAF_MPI_CALL(
      MPI_Iallreduce(MPI_IN_PLACE, msg.data(), msg.count(), msg.mpi_type(), reduce_op, comm, req));
  return cont_buf;
}

DLAF_MAKE_CALLABLE_OBJECT(allReduceInPlace);
}

template <class T>
void scheduleAllReduce(const comm::Executor& ex,
                       hpx::future<common::PromiseGuard<comm::Communicator>> pcomm, MPI_Op reduce_op,
                       hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_in,
                       hpx::future<matrix::Tile<T, Device::CPU>> tile_out) {
  using hpx::dataflow;
  using hpx::unwrapping;

  using common::internal::ContiguousBufferHolder;
  using common::internal::copyBack_o;
  using common::internal::makeItContiguous_o;
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

  hpx::future<ContiguousBufferHolder<const T>> cont_buf_in =
      dataflow(unwrapping(makeItContiguous_o), tile_in);

  hpx::future<ContiguousBufferHolder<T>> cont_buf_out;
  {
    auto wrapped = getUnwrapRetValAndArgs(
        dataflow(ex_copy, unwrapExtendTiles(makeItContiguous_o), std::move(tile_out)));
    cont_buf_out = std::move(wrapped.first);
    tile_out = std::move(hpx::get<0>(wrapped.second));
  }

  cont_buf_out = getUnwrapReturnValue(dataflow(ex, unwrapExtendTiles(internal::allReduce_o),
                                               std::move(pcomm), reduce_op, std::move(cont_buf_in),
                                               std::move(cont_buf_out), tile_in));

  dataflow(ex_copy, unwrapping(copyBack_o), std::move(cont_buf_out), std::move(tile_out));
}

template <class T>
hpx::future<matrix::Tile<T, Device::CPU>> scheduleAllReduceInPlace(
    const comm::Executor& ex, hpx::future<common::PromiseGuard<comm::Communicator>> pcomm,
    MPI_Op reduce_op, hpx::future<matrix::Tile<T, Device::CPU>> tile) {
  using hpx::dataflow;
  using hpx::unwrapping;

  using common::internal::copyBack_o;
  using common::internal::makeItContiguous_o;
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

  hpx::future<common::internal::ContiguousBufferHolder<T>> cont_buf;
  {
    auto wrapped = getUnwrapRetValAndArgs(
        dataflow(ex_copy, unwrapExtendTiles(makeItContiguous_o), std::move(tile)));
    cont_buf = std::move(wrapped.first);
    tile = std::move(hpx::get<0>(wrapped.second));
  }

  cont_buf = dataflow(ex, unwrapping(internal::allReduceInPlace_o), std::move(pcomm), reduce_op,
                      std::move(cont_buf));

  // Note:
  // This extracts the tile given as argument to copyBack, not the return value.
  return hpx::get<1>(hpx::split_future(
      dataflow(ex_copy, matrix::unwrapExtendTiles(copyBack_o), std::move(cont_buf), std::move(tile))));
}
}
}
