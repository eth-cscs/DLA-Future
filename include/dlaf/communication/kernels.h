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

#include "dlaf/common/assert.h"
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

// Non-blocking sender broadcast
template <class T, Device D>
void sendBcast(matrix::Tile<const T, D> const& tile, common::PromiseGuard<Communicator> pcomm,
               MPI_Request* req) {
  auto msg = comm::make_message(common::make_data(tile));
  MPI_Ibcast(const_cast<T*>(msg.data()), msg.count(), msg.mpi_type(), pcomm.ref().rank(), pcomm.ref(),
             req);
}

DLAF_MAKE_CALLABLE_OBJECT(sendBcast);

// Non-blocking receiver broadcast
template <class T, Device D>
matrix::Tile<T, D> recvBcast(matrix::Tile<T, D> tile, comm::IndexT_MPI root_rank,
                             common::PromiseGuard<Communicator> pcomm, MPI_Request* req) {
  auto msg = comm::make_message(common::make_data(tile));
  MPI_Ibcast(msg.data(), msg.count(), msg.mpi_type(), root_rank, pcomm.ref(), req);
  return tile;
}

DLAF_MAKE_CALLABLE_OBJECT(recvBcast);

// Non-blocking receiver broadcast (with Alloc)
template <class T, Device D>
matrix::Tile<const T, D> recvBcastAlloc(TileElementSize tile_size, comm::IndexT_MPI root_rank,
                                        common::PromiseGuard<Communicator> pcomm, MPI_Request* req) {
  using Tile_t = matrix::Tile<T, D>;
  using ConstTile_t = matrix::Tile<const T, D>;
  using MemView_t = memory::MemoryView<T, D>;

  MemView_t mem_view(tile_size.rows() * tile_size.cols());
  Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());

  auto msg = comm::make_message(common::make_data(tile));
  MPI_Ibcast(msg.data(), msg.count(), msg.mpi_type(), root_rank, pcomm.ref(), req);
  return ConstTile_t(std::move(tile));
}

template <class T, Device D, template <class> class Future>
void scheduleSendBcast(const comm::Executor& ex, Future<matrix::Tile<const T, D>> tile,
                       hpx::future<common::PromiseGuard<comm::Communicator>> pcomm) {
  hpx::dataflow(ex, hpx::util::unwrapping(sendBcast_o), internal::prepareSendTile(std::move(tile)),
                std::move(pcomm));
}

// This implements scheduleRecvBcast for when the device of the given tile and
// the communication device are different. In that case we make use of
// recvBcastAlloc, which will allocate a tile on the correct device for
// communication, receive the data, and return the tile. When communication is
// ready, we copy the data from the tile on the communication device to the tile
// passed into scheduleRecvBcast.
template <Device D, Device CommunicationD>
struct scheduleRecvBcastImpl {
  template <class T>
  static void call(const comm::Executor& ex, hpx::future<matrix::Tile<T, D>> tile,
                   comm::IndexT_MPI root_rank, hpx::future<common::PromiseGuard<Communicator>> pcomm) {
    auto tile_shared = tile.share();
    auto tile_size =
        hpx::dataflow(hpx::launch::sync,
                      hpx::util::unwrapping([](matrix::Tile<T, D> const& tile) { return tile.size(); }),
                      tile_shared);
    auto comm_tile =
        hpx::dataflow(ex, hpx::util::unwrapping(recvBcastAlloc<T, CommunicationDevice<D>::value>),
                      tile_size, root_rank, std::move(pcomm));
    hpx::dataflow(dlaf::getCopyExecutor<CommunicationDevice<D>::value, D>(),
                  matrix::unwrapExtendTiles(matrix::copy_o), std::move(comm_tile),
                  std::move(tile_shared));
  }
};

// This specialization is used when the communication device and the device used
// for the input tile are the same. In this case we don't need to do anything
// special and just launch a task that receives into the given tile.
template <Device D>
struct scheduleRecvBcastImpl<D, D> {
  template <class T>
  static void call(const comm::Executor& ex, hpx::future<matrix::Tile<T, D>> tile,
                   comm::IndexT_MPI root_rank, hpx::future<common::PromiseGuard<Communicator>> pcomm) {
    hpx::dataflow(ex, hpx::util::unwrapping(recvBcast_o), std::move(tile), root_rank, std::move(pcomm));
  }
};

template <class T, Device D>
void scheduleRecvBcast(const comm::Executor& ex, hpx::future<matrix::Tile<T, D>> tile,
                       comm::IndexT_MPI root_rank,
                       hpx::future<common::PromiseGuard<Communicator>> pcomm) {
  scheduleRecvBcastImpl<D, CommunicationDevice<D>::value>::call(ex, std::move(tile), root_rank,
                                                                std::move(pcomm));
}

template <class T, Device D>
hpx::future<matrix::Tile<const T, D>> scheduleRecvBcastAlloc(
    const comm::Executor& ex, TileElementSize tile_size, comm::IndexT_MPI root_rank,
    hpx::future<common::PromiseGuard<comm::Communicator>> pcomm) {
  return internal::handleRecvTile<D>(
      hpx::dataflow(ex, hpx::util::unwrapping(recvBcastAlloc<T, CommunicationDevice<D>::value>),
                    tile_size, root_rank, std::move(pcomm)));
}
}
}
