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
                       common::internal::ContiguousBufferHolder<T> cont_buf, MPI_Request* req) {
  auto msg = comm::make_message(cont_buf.descriptor);
  auto& comm = pcomm.ref();

  DLAF_MPI_CALL(MPI_Ireduce(MPI_IN_PLACE, msg.data(), msg.count(), msg.mpi_type(), reduce_op,
                            comm.rank(), comm, req));
  return cont_buf;
}

DLAF_MAKE_CALLABLE_OBJECT(reduceRecvInPlace);

template <class T, Device D>
auto reduceSend(comm::IndexT_MPI rank_root, common::PromiseGuard<comm::Communicator> pcomm,
                MPI_Op reduce_op, common::internal::ContiguousBufferHolder<const T> cont_buf,
                matrix::Tile<const T, D> const&, MPI_Request* req) {
  auto msg = comm::make_message(cont_buf.descriptor);
  auto& comm = pcomm.ref();

  DLAF_MPI_CALL(
      MPI_Ireduce(msg.data(), nullptr, msg.count(), msg.mpi_type(), reduce_op, rank_root, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(reduceSend);

}

template <class T, Device D>
void scheduleReduceRecvInPlace(const comm::Executor& ex,
                               hpx::future<common::PromiseGuard<comm::Communicator>> pcomm,
                               MPI_Op reduce_op, hpx::future<matrix::Tile<T, D>> tile) {
  using hpx::dataflow;
  using hpx::unwrapping;

  using common::internal::copyBack_o;
  using common::internal::makeItContiguous_o;
  using matrix::unwrapExtendTiles;

  // Note:
  //
  // TILE ---> makeContiguous --+--> CONT_BUF ----> mpi_call ---> CONT_BUF --+
  //                            |                                            |
  //                            +-------------------> TILE ------------------+-> copyBack

  auto ex_copy = getHpExecutor<Backend::MC>();

  hpx::shared_future<matrix::Tile<T, D>> tile_orig = tile.share();

  hpx::future<common::internal::ContiguousBufferHolder<T>> cont_buf;
  hpx::shared_future<matrix::Tile<T, Device::CPU>> tile_cpu = internal::prepareSendTile(tile_orig);
  cont_buf =
      matrix::getUnwrapReturnValue(dataflow(ex_copy, unwrapExtendTiles(makeItContiguous_o), tile_cpu));

  cont_buf = dataflow(ex, unwrapping(internal::reduceRecvInPlace_o), std::move(pcomm), reduce_op,
                      std::move(cont_buf));

  auto res = dataflow(ex_copy, unwrapping(copyBack_o), tile_cpu, std::move(cont_buf));

  matrix::copyIfNeeded(tile_cpu, tile_orig, std::move(res));
}

template <class T, Device D>
void scheduleReduceSend(const comm::Executor& ex, comm::IndexT_MPI rank_root,
                        hpx::future<common::PromiseGuard<comm::Communicator>> pcomm, MPI_Op reduce_op,
                        hpx::shared_future<matrix::Tile<const T, D>> tile) {
  using hpx::dataflow;
  using hpx::unwrapping;

  using common::internal::makeItContiguous_o;
  using matrix::unwrapExtendTiles;

  // Note:
  //
  // TILE -+-> makeContiguous --+--> CONT_BUF ---+--> mpi_call
  //       |                                     |
  //       +--------------> TILE ----------------+

  auto ex_copy = getHpExecutor<Backend::MC>();

  // TODO shared_future<Tile> as assumption, it requires changes for future<Tile>
  hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_cpu =
      internal::prepareSendTile(std::move(tile));
  hpx::future<common::internal::ContiguousBufferHolder<const T>> cont_buf =
      dataflow(ex_copy, unwrapping(makeItContiguous_o), tile_cpu);

  dataflow(ex, unwrapExtendTiles(internal::reduceSend_o), rank_root, std::move(pcomm), reduce_op,
           std::move(cont_buf), tile_cpu);
}
}
}
