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
#include "dlaf/common/data.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/message.h"
#include "dlaf/communication/rdma.h"
#include "dlaf/matrix/tile.h"

namespace dlaf {
namespace comm {

namespace internal {

// Note:
//
// A bag:
// - owns the temporary buffer (optionally allocated)
// - contains a data descriptor to the contiguous data to be used for communication (being it the
//    original tile or the temporary buffer)
//
// The bag can be create with the `makeItContiguous` helper function by passing the original tile
// and in case of a RW tile, it is possible to `copyBack` the memory from the temporary buffer to
// the original tile.

template <class T>
using Bag = hpx::tuple<common::Buffer<std::remove_const_t<T>>, common::DataDescriptor<T>>;

template <class T>
auto makeItContiguous(const matrix::Tile<T, Device::CPU>& tile) {
  common::Buffer<std::remove_const_t<T>> buffer;
  auto tile_data = common::make_data(tile);
  auto what_to_use = common::make_contiguous(tile_data, buffer);
  if (buffer)
    common::copy(tile_data, buffer);
  return Bag<T>(std::move(buffer), std::move(what_to_use));
}

DLAF_MAKE_CALLABLE_OBJECT(makeItContiguous);

template <class T>
auto copyBack(matrix::Tile<T, Device::CPU> tile, Bag<T> bag) {
  auto buffer_used = std::move(hpx::get<0>(bag));
  if (buffer_used)
    common::copy(buffer_used, common::make_data(tile));
  return std::move(tile);
}

DLAF_MAKE_CALLABLE_OBJECT(copyBack);

// Note:
//
// A couple of kernels for MPI collectives which represents relevant use-cases
// - reduceRecvInPlace
// - reduceSend
template <class T>
auto reduceRecvInPlace(common::PromiseGuard<comm::Communicator> pcomm, MPI_Op reduce_op,
                       internal::Bag<T> bag, MPI_Request* req) {
  auto message = comm::make_message(hpx::get<1>(bag));
  auto& communicator = pcomm.ref();

  DLAF_MPI_CALL(MPI_Ireduce(MPI_IN_PLACE, message.data(), message.count(), message.mpi_type(), reduce_op,
                            communicator.rank(), communicator, req));

  return std::move(bag);
}

DLAF_MAKE_CALLABLE_OBJECT(reduceRecvInPlace);

template <class T>
auto reduceSend(comm::IndexT_MPI rank_root, common::PromiseGuard<comm::Communicator> pcomm,
                MPI_Op reduce_op, internal::Bag<const T> bag, matrix::Tile<const T, Device::CPU> const&,
                MPI_Request* req) {
  auto message = comm::make_message(hpx::get<1>(bag));
  auto& communicator = pcomm.ref();

  DLAF_MPI_CALL(MPI_Ireduce(message.data(), nullptr, message.count(), message.mpi_type(), reduce_op,
                            rank_root, communicator, req));
}

DLAF_MAKE_CALLABLE_OBJECT(reduceSend);

}

template <class T>
auto scheduleReduceRecvInPlace(comm::Executor& ex,
                               hpx::future<common::PromiseGuard<comm::Communicator>> pcomm,
                               MPI_Op reduce_op, hpx::future<matrix::Tile<T, Device::CPU>> tile) {
  // Note:
  // Create a bag with contiguous data, and extract both:
  // - return value with the Bag
  // - extract from passed parameters the future value of the original tile passed
  //
  // The latter one is based on unwrapExtendTiles, that extends the lifetime of the future<Tile>
  // and allows to get back the ownership of the tile from the return tuple <ret_value, args>
  hpx::future<internal::Bag<T>> bag;
  {
    auto wrapped_res = hpx::split_future(
        hpx::dataflow(matrix::unwrapExtendTiles(internal::makeItContiguous_o), std::move(tile)));
    bag = std::move(hpx::get<0>(wrapped_res));
    auto args = hpx::split_future(std::move(hpx::get<1>(wrapped_res)));
    tile = std::move(hpx::get<0>(args));
  }

  // Note:
  //
  // At this point the bag is passed to the kernel, which in turns return the Bag giving back the
  // ownership of the (optional) temporary buffer, extending its lifetime till the async operation
  // is completed.
  // If the temporary buffer has not been used, the tile is not kept alive since its lifetime is not
  // extended, but it is kept alive by the next step...
  bag = hpx::dataflow(ex, hpx::util::unwrapping(internal::reduceRecvInPlace_o), std::move(pcomm),
                      reduce_op, std::move(bag));

  // Note:
  //
  // ... indeed the tile together with the future value of Bag after the async operation are "merged"
  // with the dataflow, mathing the two lifetimes.
  tile = hpx::dataflow(hpx::util::unwrapping(internal::copyBack_o), std::move(tile), std::move(bag));

  // Note:
  //
  // The returned future<Tile> is a RW tile of the original operation after the MPI Async operation
  // is fully completed (even if original tile is not used because it gets copied, it won't be
  // released)
  return std::move(tile);
}

template <class T>
auto scheduleReduceSend(comm::Executor& ex, comm::IndexT_MPI rank_root,
                        hpx::future<common::PromiseGuard<comm::Communicator>> pcomm, MPI_Op reduce_op,
                        hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile) {
  // Note:
  //
  // Similarly to what has been done in `scheduleReduceRecvInPlace`, create a Bag
  // with a temporary buffer if needed
  //
  // TODO shared_future<Tile> as assumption, it requires changes for future<Tile>
  hpx::future<internal::Bag<const T>> bag =
      hpx::dataflow(hpx::util::unwrapping(internal::makeItContiguous_o), tile);

  // Note:
  //
  // Since there is no other step after the kernel, the shared_future<Tile> has to be
  // passed to the kernel so that its lifetime gets extended during the async operation
  // execution
  hpx::dataflow(ex, hpx::util::unwrapping(internal::reduceSend_o), rank_root, std::move(pcomm),
                reduce_op, std::move(bag), tile);

  // Note:
  //
  // The returned tile is exactly the original one, because the shared ownerhsip ensures that
  // this tile won't be released before the MPI async operation is completed.
  return tile;
}

}
}
