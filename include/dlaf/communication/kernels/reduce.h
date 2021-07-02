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

#include "dlaf/common/bag.h"
#include "dlaf/common/callable_object.h"
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
                       common::internal::Bag<T> bag, MPI_Request* req) {
  auto msg = comm::make_message(hpx::get<1>(bag));
  auto& comm = pcomm.ref();

  DLAF_MPI_CALL(MPI_Ireduce(MPI_IN_PLACE, msg.data(), msg.count(), msg.mpi_type(), reduce_op,
                            comm.rank(), comm, req));

  return bag;
}

DLAF_MAKE_CALLABLE_OBJECT(reduceRecvInPlace);

template <class T>
auto reduceSend(comm::IndexT_MPI rank_root, common::PromiseGuard<comm::Communicator> pcomm,
                MPI_Op reduce_op, common::internal::Bag<const T> bag,
                matrix::Tile<const T, Device::CPU> const&, MPI_Request* req) {
  auto msg = comm::make_message(hpx::get<1>(bag));
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
  // Note:
  //
  // TILE ---> makeContiguous --+--> BAG ----> mpi_call ---> BAG --+
  //                            |                                  |
  //                            +-------------> TILE --------------+-> copyBack

  hpx::future<common::internal::Bag<T>> bag;
  {
    // clang-format off
    auto wrapped = getUnwrapRetValAndArgs(
        hpx::dataflow(
          dlaf::getHpExecutor<Backend::MC>(),
          matrix::unwrapExtendTiles(common::internal::makeItContiguous_o),
          std::move(tile)));
    // clang-format on

    bag = std::move(wrapped.first);
    tile = std::move(hpx::get<0>(wrapped.second));
  }

  // clang-format off
  bag = hpx::dataflow(
      ex,
      hpx::util::unwrapping(internal::reduceRecvInPlace_o),
      std::move(pcomm), reduce_op, std::move(bag));
  // clang-format on

  // clang-format off
  hpx::dataflow(
      dlaf::getHpExecutor<Backend::MC>(),
      hpx::util::unwrapping(common::internal::copyBack_o),
      std::move(tile), std::move(bag));
  // clang-format on
}

template <class T>
void scheduleReduceSend(const comm::Executor& ex, comm::IndexT_MPI rank_root,
                        hpx::future<common::PromiseGuard<comm::Communicator>> pcomm, MPI_Op reduce_op,
                        hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile) {
  // Note:
  //
  // TILE ---> makeContiguous --+--> BAG ---+--> mpi_call
  //                            |           |
  //                            +--> TILE --+

  // TODO shared_future<Tile> as assumption, it requires changes for future<Tile>
  // clang-format off
  hpx::future<common::internal::Bag<const T>> bag =
      hpx::dataflow(
          dlaf::getHpExecutor<Backend::MC>(),
          hpx::util::unwrapping(common::internal::makeItContiguous_o),
          tile);
  // clang-format on

  // clang-format off
  hpx::dataflow(
      ex,
      matrix::unwrapExtendTiles(internal::reduceSend_o),
      rank_root, std::move(pcomm), reduce_op, std::move(bag), tile);
  // clang-format on
}
}
}
