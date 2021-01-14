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

#include <hpx/include/parallel_executors.hpp>
#include <hpx/local/future.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/common/data.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/message.h"
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

template <class T>
void send_tile(common::Pipeline<comm::CommunicatorGrid>& task_chain, Coord rc_comm,
               hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile) {
  using ConstTile_t = matrix::Tile<const T, Device::CPU>;
  using PromiseComm_t = common::PromiseGuard<comm::CommunicatorGrid>;

  auto send_bcast_f = hpx::util::annotated_function(
      [rc_comm](hpx::shared_future<ConstTile_t> ftile, hpx::future<PromiseComm_t> fpcomm) {
        PromiseComm_t pcomm = fpcomm.get();
        comm::sync::broadcast::send(pcomm.ref().subCommunicator(rc_comm), ftile.get());
      },
      "send_tile");
  hpx::dataflow(std::move(send_bcast_f), tile, task_chain());
}

template <class T>
void send_tile(hpx::threads::executors::pool_executor ex,
               common::Pipeline<comm::CommunicatorGrid>& task_chain, Coord rc_comm,
               hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile) {
  using ConstTile_t = matrix::Tile<const T, Device::CPU>;
  using PromiseComm_t = common::PromiseGuard<comm::CommunicatorGrid>;

  auto send_bcast_f = hpx::util::annotated_function(
      [rc_comm](hpx::shared_future<ConstTile_t> ftile, hpx::future<PromiseComm_t> fpcomm) {
        PromiseComm_t pcomm = fpcomm.get();
        comm::sync::broadcast::send(pcomm.ref().subCommunicator(rc_comm), ftile.get());
      },
      "send_tile");
  hpx::dataflow(ex, std::move(send_bcast_f), tile, task_chain());
}

template <class T>
void recv_tile(common::Pipeline<comm::CommunicatorGrid>& mpi_task_chain, Coord rc_comm,
               hpx::future<matrix::Tile<T, Device::CPU>> tile, int rank) {
  using PromiseComm_t = common::PromiseGuard<comm::CommunicatorGrid>;
  using Tile_t = matrix::Tile<T, Device::CPU>;

  auto recv_bcast_f = hpx::util::annotated_function(
      [rank, rc_comm](hpx::future<PromiseComm_t> fpcomm, hpx::future<Tile_t> ftile) {
        PromiseComm_t pcomm = fpcomm.get();
        Tile_t tile = ftile.get();
        comm::sync::broadcast::receive_from(rank, pcomm.ref().subCommunicator(rc_comm), tile);
      },
      "recv_tile");
  hpx::dataflow(std::move(recv_bcast_f), mpi_task_chain(), std::move(tile));
}

template <class T>
hpx::future<matrix::Tile<const T, Device::CPU>> recv_tile(
    hpx::execution::parallel_executor ex, common::Pipeline<comm::CommunicatorGrid>& mpi_task_chain,
    Coord rc_comm, TileElementSize tile_size, int rank) {
  using ConstTile_t = matrix::Tile<const T, Device::CPU>;
  using PromiseComm_t = common::PromiseGuard<comm::CommunicatorGrid>;
  using MemView_t = memory::MemoryView<T, Device::CPU>;
  using Tile_t = matrix::Tile<T, Device::CPU>;

  auto recv_bcast_f = hpx::util::annotated_function(
      [rank, tile_size, rc_comm](hpx::future<PromiseComm_t> fpcomm) -> ConstTile_t {
        PromiseComm_t pcomm = fpcomm.get();
        MemView_t mem_view(tile_size.linear_size());
        Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());
        comm::sync::broadcast::receive_from(rank, pcomm.ref().subCommunicator(rc_comm), tile);
        return ConstTile_t(std::move(tile));
      },
      "recv_tile");
  return hpx::dataflow(ex, std::move(recv_bcast_f), mpi_task_chain());
}
}
}
