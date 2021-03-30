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

#include <hpx/local/future.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/common/callable_object.h"
#include "dlaf/common/data.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/device.h"
#include "dlaf/communication/message.h"
#include "dlaf/executors.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/tile.h"

namespace dlaf {
namespace comm {
namespace sync {
namespace broadcast {

/// MPI_Bcast wrapper for sender side accepting a Data.
///
/// For more information, see the Data concept in "dlaf/common/data.h".
template <class DataIn>
void send(Communicator& communicator, DataIn&& message_to_send) {
  auto data = common::make_data(message_to_send);
  using DataT = std::remove_const_t<typename common::data_traits<decltype(data)>::element_t>;

  auto message = comm::make_message(std::move(data));
  DLAF_MPI_CALL(MPI_Bcast(const_cast<DataT*>(message.data()), message.count(), message.mpi_type(),
                          communicator.rank(), communicator));
}

/// MPI_Bcast wrapper for receiver side accepting a dlaf::comm::Message.
///
/// For more information, see the Data concept in "dlaf/common/data.h".
template <class DataOut>
void receive_from(const int broadcaster_rank, Communicator& communicator, DataOut&& data) {
  DLAF_ASSERT_HEAVY(broadcaster_rank != communicator.rank(), broadcaster_rank, communicator.rank());
  auto message = comm::make_message(common::make_data(std::forward<DataOut>(data)));
  DLAF_MPI_CALL(
      MPI_Bcast(message.data(), message.count(), message.mpi_type(), broadcaster_rank, communicator));
}
}
}

namespace internal {
/// Helper function for preparing a tile for sending.
///
/// Duplicates the tile to CPU memory if CUDA RDMA is not enabled for MPI.
/// Returns the tile unmodified otherwise.
template <Device D, typename T>
auto prepareSendTile(hpx::shared_future<matrix::Tile<const T, D>> tile) {
  return matrix::duplicateIfNeeded<CommunicationDevice<D>::value>(std::move(tile));
}

/// Helper function for handling a tile after receiving.
///
/// If CUDA RDMA is disabled, the tile returned from recvTile will always be on
/// the CPU. This helper duplicates to the GPU if the first template parameter
/// is a GPU device. The first template parameter must be given.
template <Device D, typename T>
auto handleRecvTile(hpx::future<matrix::Tile<const T, CommunicationDevice<D>::value>> tile) {
  return matrix::duplicateIfNeeded<D>(std::move(tile));
}
}

/// Task for broadcasting (send endpoint) a Tile in a direction over a CommunicatorGrid
template <typename T, Device D, template <class> class Future>
void sendTile(hpx::future<common::PromiseGuard<comm::CommunicatorGrid>> mpi_task_chain, Coord rc_comm,
              Future<matrix::Tile<const T, D>> tile) {
  using PromiseComm_t = common::PromiseGuard<comm::CommunicatorGrid>;

  PromiseComm_t pcomm = mpi_task_chain.get();
  comm::sync::broadcast::send(pcomm.ref().subCommunicator(rc_comm), tile.get());
}

DLAF_MAKE_CALLABLE_OBJECT(sendTile);

/// Task for broadcasting (receiving endpoint) a Tile in a direction over a CommunicatorGrid
template <class T, Device D>
void recvTile(hpx::future<common::PromiseGuard<comm::CommunicatorGrid>> mpi_task_chain, Coord rc_comm,
              hpx::future<matrix::Tile<T, CommunicationDevice<D>::value>> tile, comm::IndexT_MPI rank) {
  using PromiseComm_t = common::PromiseGuard<comm::CommunicatorGrid>;

  PromiseComm_t pcomm = mpi_task_chain.get();
  comm::sync::broadcast::receive_from(rank, pcomm.ref().subCommunicator(rc_comm), tile.get());
}

DLAF_MAKE_CALLABLE_OBJECT(recvTile);

/// Task for broadcasting (receiving endpoint) a Tile ("JIT" allocation) in a direction over a CommunicatorGrid
template <class T, Device D>
matrix::Tile<const T, CommunicationDevice<D>::value> recvAllocTile(
    hpx::future<common::PromiseGuard<comm::CommunicatorGrid>> mpi_task_chain, Coord rc_comm,
    TileElementSize tile_size, comm::IndexT_MPI rank) {
  constexpr Device comm_device = CommunicationDevice<D>::value;
  using ConstTile_t = matrix::Tile<const T, comm_device>;
  using PromiseComm_t = common::PromiseGuard<comm::CommunicatorGrid>;
  using MemView_t = memory::MemoryView<T, comm_device>;
  using Tile_t = matrix::Tile<T, comm_device>;

  PromiseComm_t pcomm = mpi_task_chain.get();
  MemView_t mem_view(tile_size.linear_size());
  Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());
  comm::sync::broadcast::receive_from(rank, pcomm.ref().subCommunicator(rc_comm), tile);
  return ConstTile_t(std::move(tile));
}
}
}
