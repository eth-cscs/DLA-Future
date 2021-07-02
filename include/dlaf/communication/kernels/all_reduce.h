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
               common::internal::Bag<const T> bag_in, common::internal::Bag<T> bag_out,
               matrix::Tile<const T, Device::CPU> const&, MPI_Request* req) {
  auto& comm = pcomm.ref();
  auto msg_in = comm::make_message(hpx::get<1>(bag_in));
  auto msg_out = comm::make_message(hpx::get<1>(bag_out));

  DLAF_MPI_CALL(MPI_Iallreduce(msg_in.data(), msg_out.data(), msg_in.count(), msg_in.mpi_type(),
                               reduce_op, comm, req));

  return bag_out;
}

DLAF_MAKE_CALLABLE_OBJECT(allReduce);

template <class T>
auto allReduceInPlace(common::PromiseGuard<comm::Communicator> pcomm, MPI_Op reduce_op,
                      common::internal::Bag<T> bag, MPI_Request* req) {
  auto& comm = pcomm.ref();
  auto msg = comm::make_message(hpx::get<1>(bag));

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
  // Note:
  //
  //         +---------------------------------+
  //         |                                 |
  // TILE_I -+-> makeContiguous -----> BAG_I --+--> mpi_call --> BAG_O --+
  //                                           |                         |
  // TILE_O ---> makeContiguous --+--> BAG_O --+                         |
  //                              |                                      |
  //                              +----------------> TILE_O -------------+-> copyBack

  hpx::future<common::internal::Bag<const T>> bag_in =
      hpx::dataflow(hpx::util::unwrapping(common::internal::makeItContiguous_o), tile_in);

  hpx::future<common::internal::Bag<T>> bag_out;
  {
    // clang-format off
    auto wrapped = getUnwrapRetValAndArgs(
        hpx::dataflow(
          dlaf::getHpExecutor<Backend::MC>(),
          matrix::unwrapExtendTiles(common::internal::makeItContiguous_o),
          std::move(tile_out)));
    // clang-format on
    bag_out = std::move(wrapped.first);
    tile_out = std::move(hpx::get<0>(wrapped.second));
  }

  // clang-format off
  bag_out = getUnwrapReturnValue(hpx::dataflow(
      ex,
      matrix::unwrapExtendTiles(internal::allReduce_o),
      std::move(pcomm), reduce_op, std::move(bag_in), std::move(bag_out), tile_in));
  // clang-format on

  // clang-format off
  hpx::dataflow(
      dlaf::getHpExecutor<Backend::MC>(),
      hpx::util::unwrapping(common::internal::copyBack_o),
      std::move(tile_out), std::move(bag_out));
  // clang-format on
}

template <class T>
hpx::future<matrix::Tile<T, Device::CPU>> scheduleAllReduceInPlace(
    const comm::Executor& ex, hpx::future<common::PromiseGuard<comm::Communicator>> pcomm,
    MPI_Op reduce_op, hpx::future<matrix::Tile<T, Device::CPU>> tile) {
  // Note:
  //
  // TILE ---> makeContiguous --+--> BAG ----> mpi_call ---> BAG --+
  //                            |                                  |
  //                            +-------------> TILE --------------+-> copyBack ---> TILE
  //
  //
  // The last TILE after the copyBack is returned so that other task can be attached to it,
  // AFTER the asynchronous MPI_AllReduce has completed

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
      hpx::util::unwrapping(internal::allReduceInPlace_o),
      std::move(pcomm), reduce_op, std::move(bag));
  // clang-format on

  // clang-format off
  return hpx::dataflow(
      dlaf::getHpExecutor<Backend::MC>(),
      hpx::util::unwrapping(common::internal::copyBack_o),
      std::move(tile), std::move(bag));
  // clang-format on
}
}
}
