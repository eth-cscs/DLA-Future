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
  auto& communicator = pcomm.ref();
  auto message_in = comm::make_message(hpx::get<1>(bag_in));
  auto message_out = comm::make_message(hpx::get<1>(bag_out));

  DLAF_MPI_CALL(MPI_Iallreduce(message_in.data(), message_out.data(), message_in.count(),
                               message_in.mpi_type(), reduce_op, communicator, req));

  return std::move(bag_out);
}

DLAF_MAKE_CALLABLE_OBJECT(allReduce);

template <class T>
auto allReduceInPlace(common::PromiseGuard<comm::Communicator> pcomm, MPI_Op reduce_op,
                      common::internal::Bag<T> bag, MPI_Request* req) {
  auto& communicator = pcomm.ref();
  auto message = comm::make_message(hpx::get<1>(bag));

  DLAF_MPI_CALL(MPI_Iallreduce(MPI_IN_PLACE, message.data(), message.count(), message.mpi_type(),
                               reduce_op, communicator, req));

  return std::move(bag);
}

DLAF_MAKE_CALLABLE_OBJECT(allReduceInPlace);
}

template <class T>
void scheduleAllReduce(const comm::Executor& ex,
                       hpx::future<common::PromiseGuard<comm::Communicator>> pcomm, MPI_Op reduce_op,
                       hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_in,
                       hpx::future<matrix::Tile<T, Device::CPU>> tile_out) {
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
    bag_out = std::move(hpx::get<0>(wrapped));
    tile_out = std::move(hpx::get<0>(std::move(hpx::get<1>(wrapped))));
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
auto scheduleAllReduceInPlace(const comm::Executor& ex,
                              hpx::future<common::PromiseGuard<comm::Communicator>> pcomm,
                              MPI_Op reduce_op, hpx::future<matrix::Tile<T, Device::CPU>> tile) {
  hpx::future<common::internal::Bag<T>> bag;
  {
    // clang-format off
    auto wrapped = getUnwrapRetValAndArgs(
        hpx::dataflow(
          dlaf::getHpExecutor<Backend::MC>(),
          matrix::unwrapExtendTiles(common::internal::makeItContiguous_o),
          std::move(tile)));
    // clang-format on
    bag = std::move(hpx::get<0>(wrapped));
    tile = std::move(hpx::get<0>(std::move(hpx::get<1>(wrapped))));
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
