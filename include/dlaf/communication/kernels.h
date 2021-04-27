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
matrix::Tile<T, D> recvBcast(matrix::Tile<T, D> tile, int root_rank,
                             common::PromiseGuard<Communicator> pcomm, MPI_Request* req) {
  auto msg = comm::make_message(common::make_data(tile));
  MPI_Ibcast(msg.data(), msg.count(), msg.mpi_type(), root_rank, pcomm.ref(), req);
  return std::move(tile);
}

DLAF_MAKE_CALLABLE_OBJECT(recvBcast);

// Non-blocking receiver broadcast (with Alloc)
template <class T, Device D>
matrix::Tile<const T, D> recvBcastAlloc(TileElementSize tile_size, comm::IndexT_MPI root_rank,
                                        common::PromiseGuard<Communicator> pcomm, MPI_Request* req) {
  using Tile_t = matrix::Tile<T, D>;
  using ConstTile_t = matrix::Tile<const T, D>;
  using MemView_t = memory::MemoryView<T, D>;

  MemView_t mem_view(util::size_t::mul(tile_size.rows(), tile_size.cols()));
  Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());

  auto msg = comm::make_message(common::make_data(tile));
  MPI_Ibcast(msg.data(), msg.count(), msg.mpi_type(), root_rank, pcomm.ref(), req);
  return ConstTile_t(std::move(tile));
}

template <class T, Device D, class Executor, template <class> class Future>
void scheduleSendBcast(Executor&& ex, Future<matrix::Tile<const T, D>> tile,
                       hpx::future<common::PromiseGuard<comm::Communicator>> pcomm) {
  hpx::dataflow(std::forward<Executor>(ex), hpx::util::unwrapping(sendBcast_o),
                internal::prepareSendTile(std::move(tile)), std::move(pcomm));
}

template <class T, Device D, class Executor>
hpx::future<matrix::Tile<T, D>> scheduleRecvBcast(Executor&& ex, hpx::future<matrix::Tile<T, D>> tile,
                                                  comm::IndexT_MPI root_rank,
                                                  hpx::future<common::PromiseGuard<Communicator>> pcomm) {
  return internal::handleRecvTile<D>(hpx::dataflow(std::forward<Executor>(ex),
                                                   hpx::util::unwrapping(recvBcast_o), std::move(tile),
                                                   root_rank, std::move(pcomm)));
}

template <class T, Device D, class Executor>
hpx::future<matrix::Tile<const T, D>> scheduleRecvBcastAlloc(
    Executor&& ex, TileElementSize tile_size, int root_rank,
    hpx::future<common::PromiseGuard<comm::Communicator>> pcomm) {
  return internal::handleRecvTile<D>(
      hpx::dataflow(std::forward<Executor>(ex),
                    hpx::util::unwrapping(recvBcastAlloc<T, CommunicationDevice<D>::value>), tile_size,
                    root_rank, std::move(pcomm)));
}
}
}
