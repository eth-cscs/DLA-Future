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
#include "dlaf/communication/message.h"
#include "dlaf/executors.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/tile.h"

namespace dlaf {
namespace comm {

template <Device D>
struct CommunicationDevice {
  static constexpr Device value = D;
};

#if defined(DLAF_WITH_CUDA) && !defined(DLAF_WITH_CUDA_RDMA)
template <>
struct CommunicationDevice<Device::GPU> {
  static constexpr Device value = Device::CPU;
};
#endif

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

namespace detail {
template <Device Source, Device Destination>
struct DuplicateIfNeeded {
  template <typename T>
  static auto call(hpx::future<matrix::Tile<const T, Source>> tile) {
    return dlaf::matrix::getReturnValue(
        hpx::dataflow(getCopyExecutor<Source, Destination>(),
                      dlaf::matrix::unwrapExtendTiles(dlaf::matrix::Duplicate<const T, Destination>{}),
                      tile));
  }

  template <typename T>
  static auto call(hpx::shared_future<matrix::Tile<const T, Source>> tile) {
    return dlaf::matrix::getReturnValue(
        hpx::dataflow(getCopyExecutor<Source, Destination>(),
                      dlaf::matrix::unwrapExtendTiles(dlaf::matrix::Duplicate<const T, Destination>{}),
                      tile));
  }
};

template <Device SourceDestination>
struct DuplicateIfNeeded<SourceDestination, SourceDestination> {
  template <typename T>
  static auto call(hpx::future<matrix::Tile<const T, SourceDestination>> tile) {
    return tile;
  }

  template <typename T>
  static auto call(hpx::shared_future<matrix::Tile<const T, SourceDestination>> tile) {
    return tile;
  }
};
}

template <typename T, Device D>
auto prepareSendTile(hpx::shared_future<matrix::Tile<const T, D>> tile) {
  return detail::DuplicateIfNeeded<D, CommunicationDevice<D>::value>::call(std::move(tile));
}

template <Device D, typename T>
auto handleRecvTile(hpx::future<matrix::Tile<const T, CommunicationDevice<D>::value>> tile) {
  return detail::DuplicateIfNeeded<CommunicationDevice<D>::value, D>::call(std::move(tile));
}

/// Task for broadcasting (send endpoint) a Tile in a direction over a CommunicatorGrid
template <class TileFuture>
void sendTile(hpx::future<common::PromiseGuard<comm::CommunicatorGrid>> mpi_task_chain, Coord rc_comm,
              TileFuture&& tile) {
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
