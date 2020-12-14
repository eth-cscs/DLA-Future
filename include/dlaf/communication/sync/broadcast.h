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

#if defined(DLAF_WITH_CUDA) && !defined(DLAF_WITH_CUDA_RDMA)
template <class T, Device D, class Executor>
void send_tile(Executor&& ex, common::Pipeline<comm::CommunicatorGrid>& task_chain, Coord rc_comm,
               hpx::shared_future<matrix::Tile<const T, D>> tile) {
  using ConstTile_t = matrix::Tile<const T, D>;
  using CPUMemView_t = memory::MemoryView<T, Device::CPU>;
  using CPUConstTile_t = matrix::Tile<T, Device::CPU>;
  using CPUTile_t = matrix::Tile<T, Device::CPU>;
  using PromiseComm_t = common::PromiseGuard<comm::CommunicatorGrid>;

  auto deep_copy_f = hpx::util::annotated_function(
      [](hpx::shared_future<ConstTile_t> ftile, auto&&... ts) {
        auto tile_size = ftile.get().size();
        CPUMemView_t mem_view(tile_size.linear_size());
        CPUTile_t tile(tile_size, std::move(mem_view), tile_size.rows());
        dlaf::matrix::copy(ftile.get(), tile, std::forward<decltype(ts)>(ts)...);
        return CPUConstTile_t(std::move(tile));
      },
      "copy_tile_to_host");

  auto send_bcast_f = hpx::util::annotated_function(
      [rc_comm](hpx::shared_future<CPUConstTile_t> ftile, hpx::future<PromiseComm_t> fpcomm) {
        PromiseComm_t pcomm = fpcomm.get();
        comm::sync::broadcast::send(pcomm.ref().subCommunicator(rc_comm), ftile.get());
      },
      "send_tile");

  auto cpu_tile = hpx::dataflow(getCopyExecutor<D, Device::CPU>(), std::move(deep_copy_f), tile);
  hpx::dataflow(std::forward<Executor>(ex), std::move(send_bcast_f), cpu_tile, task_chain());
}

template <class T, Device D, class Executor>
hpx::future<matrix::Tile<const T, D>> recv_tile(
    Executor&& ex, common::Pipeline<comm::CommunicatorGrid>& mpi_task_chain, Coord rc_comm,
    TileElementSize tile_size, int rank) {
  using ConstTile_t = matrix::Tile<const T, D>;
  using CPUMemView_t = memory::MemoryView<T, Device::CPU>;
  using CPUConstTile_t = matrix::Tile<T, Device::CPU>;
  using CPUTile_t = matrix::Tile<T, Device::CPU>;
  using PromiseComm_t = common::PromiseGuard<comm::CommunicatorGrid>;
  using MemView_t = memory::MemoryView<T, D>;
  using Tile_t = matrix::Tile<T, D>;

  auto recv_bcast_f = hpx::util::annotated_function(
      [rank, tile_size, rc_comm](hpx::future<PromiseComm_t> fpcomm) -> CPUConstTile_t {
        PromiseComm_t pcomm = fpcomm.get();
        CPUTile_t tile(tile_size, CPUMemView_t(tile_size.linear_size()), tile_size.rows());
        comm::sync::broadcast::receive_from(rank, pcomm.ref().subCommunicator(rc_comm), tile);
        return CPUConstTile_t(std::move(tile));
      },
      "recv_tile");

  auto deep_copy_f = hpx::util::annotated_function(
      [](hpx::shared_future<CPUConstTile_t> ftile, auto&&... ts) {
        auto tile_size = ftile.get().size();
        MemView_t mem_view(tile_size.linear_size() * 2);
        Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());
        dlaf::matrix::copy(ftile.get(), tile, std::forward<decltype(ts)>(ts)...);
        // Have to make sure the CPU tile lives until copy is completed
        return hpx::make_tuple(ConstTile_t(std::move(tile)), ftile);
      },
      "copy_tile_to_device");

  auto cpu_tile = hpx::dataflow(std::forward<Executor>(ex), std::move(recv_bcast_f), mpi_task_chain());
  auto gpu_cpu_tile = hpx::dataflow(getCopyExecutor<Device::CPU, D>(), std::move(deep_copy_f), cpu_tile);
  auto split_tile = hpx::split_future(std::move(gpu_cpu_tile));
  auto gpu_tile = std::move(hpx::get<0>(split_tile));
  return gpu_tile;
}

#else

template <class T, Device D, class Executor>
void send_tile(Executor&& ex, common::Pipeline<comm::CommunicatorGrid>& task_chain, Coord rc_comm,
               hpx::shared_future<matrix::Tile<const T, D>> tile) {
  using ConstTile_t = matrix::Tile<const T, D>;
  using PromiseComm_t = common::PromiseGuard<comm::CommunicatorGrid>;

  PromiseComm_t pcomm = mpi_task_chain.get();
  comm::sync::broadcast::receive_from(rank, pcomm.ref().subCommunicator(rc_comm), tile.get());
}

template <class T, Device D, class Executor>
hpx::future<matrix::Tile<const T, D>> recv_tile(
    Executor&& ex, common::Pipeline<comm::CommunicatorGrid>& mpi_task_chain, Coord rc_comm,
    TileElementSize tile_size, int rank) {
  using ConstTile_t = matrix::Tile<const T, D>;
  using PromiseComm_t = common::PromiseGuard<comm::CommunicatorGrid>;
  using MemView_t = memory::MemoryView<T, D>;
  using Tile_t = matrix::Tile<T, D>;

  PromiseComm_t pcomm = mpi_task_chain.get();
  MemView_t mem_view(tile_size.linear_size());
  Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());
  comm::sync::broadcast::receive_from(rank, pcomm.ref().subCommunicator(rc_comm), tile);
  return ConstTile_t(std::move(tile));
}
#endif
}
}
