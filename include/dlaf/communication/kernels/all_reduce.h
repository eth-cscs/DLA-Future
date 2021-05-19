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

}

template <class T>
auto scheduleAllReduce(const comm::Executor& ex,
                       hpx::future<common::PromiseGuard<comm::Communicator>> pcomm, MPI_Op reduce_op,
                       hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_in,
                       hpx::future<matrix::Tile<T, Device::CPU>> tile_out) {
  hpx::future<common::internal::Bag<const T>> bag_in =
      hpx::dataflow(hpx::util::unwrapping(common::internal::makeItContiguous_o), tile_in);

  hpx::future<common::internal::Bag<T>> bag_out;
  {
    auto wrapped = getUnwrapRetValAndArgs(
        hpx::dataflow(matrix::unwrapExtendTiles(common::internal::makeItContiguous_o),
                      std::move(tile_out)));
    bag_out = std::move(hpx::get<0>(wrapped));
    tile_out = std::move(hpx::get<0>(std::move(hpx::get<1>(wrapped))));
  }

  bag_out = hpx::dataflow(ex, hpx::util::unwrapping(internal::allReduce_o), std::move(pcomm), reduce_op,
                          std::move(bag_in), std::move(bag_out), tile_in);

  tile_out = hpx::dataflow(hpx::util::unwrapping(common::internal::copyBack_o), std::move(tile_out),
                           std::move(bag_out));

  return tile_out;
}

}
}
