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

template <class T>
auto reduceSend(comm::IndexT_MPI rank_root, common::PromiseGuard<comm::Communicator> pcomm,
                MPI_Op reduce_op, common::internal::ContiguousBufferHolder<const T> cont_buf,
                matrix::Tile<const T, Device::CPU> const&, MPI_Request* req) {
  auto msg = comm::make_message(cont_buf.descriptor);
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

  hpx::future<common::internal::ContiguousBufferHolder<T>> cont_buf;
  {
    auto wrapped = getUnwrapRetValAndArgs(
        dataflow(ex_copy, unwrapExtendTiles(hpx::util::annotated_function(makeItContiguous_o, "load")),
                 std::move(tile)));

    cont_buf = std::move(wrapped.first);
    tile = std::move(hpx::get<0>(wrapped.second));
  }

  cont_buf =
      dataflow(ex, unwrapping(hpx::util::annotated_function(internal::reduceRecvInPlace_o, "recv")),
               std::move(pcomm), reduce_op, std::move(cont_buf));

  dataflow(ex_copy, unwrapping(copyBack_o), std::move(tile), std::move(cont_buf));
}

template <class T>
void scheduleReduceSend(const comm::Executor& ex, comm::IndexT_MPI rank_root,
                        hpx::future<common::PromiseGuard<comm::Communicator>> pcomm, MPI_Op reduce_op,
                        hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile) {
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
  hpx::future<common::internal::ContiguousBufferHolder<const T>> cont_buf =
      dataflow(ex_copy, unwrapping(hpx::util::annotated_function(makeItContiguous_o, "offload")), tile);

  dataflow(ex, unwrapExtendTiles(hpx::util::annotated_function(internal::reduceSend_o, "send")),
           rank_root, std::move(pcomm), reduce_op, std::move(cont_buf), tile);
}
}
}
