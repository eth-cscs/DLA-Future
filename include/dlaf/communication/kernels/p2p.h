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

// Non-blocking point to point send
template <class T, Device D>
void send(const matrix::Tile<const T, D>& tile, IndexT_MPI receiver, IndexT_MPI tag,
          common::PromiseGuard<Communicator> pcomm, MPI_Request* req) {
  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CALL(MPI_Isend(const_cast<T*>(msg.data()), msg.count(), msg.mpi_type(), receiver, tag,
                          pcomm.ref(), req));
}

DLAF_MAKE_CALLABLE_OBJECT(send);

// Non-blocking point to point receive
template <class T, Device D>
matrix::Tile<const T, D> recvAlloc(TileElementSize tile_size, IndexT_MPI sender, IndexT_MPI tag,
                                   common::PromiseGuard<Communicator> pcomm, MPI_Request* req) {
  using Tile_t = matrix::Tile<T, D>;
  using ConstTile_t = matrix::Tile<const T, D>;
  using MemView_t = memory::MemoryView<T, D>;

  MemView_t mem_view(tile_size.rows() * tile_size.cols());
  Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CALL(MPI_Irecv(msg.data(), msg.count(), msg.mpi_type(), sender, tag, pcomm.ref(), req));
  return ConstTile_t(std::move(tile));
}

template <class T, Device D, class Executor, template <class> class Future>
void scheduleSend(Executor&& ex, Future<matrix::Tile<const T, D>> tile, IndexT_MPI receiver,
                  IndexT_MPI tag, hpx::future<common::PromiseGuard<comm::Communicator>> pcomm) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(send_o),
                internal::prepareSendTile(std::move(tile)), receiver, tag, std::move(pcomm));
}

template <class T, Device D, class Executor>
hpx::future<matrix::Tile<const T, D>> scheduleRecvAlloc(
    Executor&& ex, TileElementSize tile_size, IndexT_MPI sender, IndexT_MPI tag,
    hpx::future<common::PromiseGuard<comm::Communicator>> pcomm) {
  return internal::handleRecvTile<D>(
      hpx::dataflow(std::forward<Executor>(ex),
                    hpx::unwrapping(recvAlloc<T, CommunicationDevice<D>::value>), tile_size, sender, tag,
                    std::move(pcomm)));
}
}
}
