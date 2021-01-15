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

namespace detail {
template <Device D>
struct prepare_send_tile {
  template <typename T>
  static auto call(hpx::shared_future<matrix::Tile<const T, D>> tile) {
    return tile;
  }
};

#if defined(DLAF_WITH_CUDA) && !defined(DLAF_WITH_CUDA_RDMA)
template <>
struct prepare_send_tile<Device::GPU> {
  template <typename T>
  static auto call(hpx::shared_future<matrix::Tile<const T, Device::GPU>> tile) {
    // TODO: Nicer API for Duplicate?
    auto gpu_cpu_tile =
        hpx::dataflow(getCopyExecutor<Device::GPU, Device::CPU>(),
                      dlaf::matrix::unwrapExtendTiles(dlaf::matrix::Duplicate<const T, Device::CPU>{}),
                      tile);
    // TODO: Nicer API for getting only the result?
    auto split_tile = hpx::split_future(std::move(gpu_cpu_tile));
    return std::move(hpx::get<0>(split_tile));
  }
};
#endif

template <Device DOut>
struct handle_recv_tile {
  template <typename T, Device D>
  static auto call(hpx::future<matrix::Tile<const T, D>> tile) {
    return tile;
  }
};

#if defined(DLAF_WITH_CUDA) && !defined(DLAF_WITH_CUDA_RDMA)
template <>
struct handle_recv_tile<Device::GPU> {
  template <typename T>
  static auto call(hpx::future<matrix::Tile<const T, Device::CPU>> tile) {
    auto gpu_cpu_tile =
        hpx::dataflow(getCopyExecutor<Device::CPU, Device::GPU>(),
                      dlaf::matrix::unwrapExtendTiles(dlaf::matrix::Duplicate<const T, Device::GPU>{}),
                      tile);
    auto split_tile = hpx::split_future(std::move(gpu_cpu_tile));
    return std::move(hpx::get<0>(split_tile));
  }
};
#endif
}

template <class T, Device D>
void send_tile(hpx::execution::parallel_executor ex, common::Pipeline<comm::CommunicatorGrid>& task_chain, Coord rc_comm,
               hpx::shared_future<matrix::Tile<const T, D>> tile) {
  using PromiseComm_t = common::PromiseGuard<comm::CommunicatorGrid>;

  auto send_bcast_f = hpx::util::annotated_function(
      [rc_comm](auto ftile, hpx::future<PromiseComm_t> fpcomm) {
        PromiseComm_t pcomm = fpcomm.get();
        comm::sync::broadcast::send(pcomm.ref().subCommunicator(rc_comm), ftile.get());
      },
      "send_tile");

  hpx::dataflow(ex, std::move(send_bcast_f),
                detail::prepare_send_tile<D>::call(std::move(tile)), task_chain());
}

template <class T, Device D>
hpx::future<matrix::Tile<const T, D>> recv_tile(hpx::execution::parallel_executor ex,
                                                common::Pipeline<comm::CommunicatorGrid>& mpi_task_chain,
                                                Coord rc_comm, TileElementSize tile_size, int rank) {
  using PromiseComm_t = common::PromiseGuard<comm::CommunicatorGrid>;

#if defined(DLAF_WITH_CUDA) && !defined(DLAF_WITH_CUDA_RDMA)
  constexpr Device device = Device::CPU;
#else
  constexpr Device device = D;
#endif
  using ConstTile_t = matrix::Tile<const T, device>;
  using MemView_t = memory::MemoryView<T, device>;
  using Tile_t = matrix::Tile<T, device>;

  auto recv_bcast_f = hpx::util::annotated_function(
      [rank, tile_size, rc_comm](hpx::future<PromiseComm_t> fpcomm) -> ConstTile_t {
        PromiseComm_t pcomm = fpcomm.get();
        Tile_t tile(tile_size, MemView_t(tile_size.linear_size()), tile_size.rows());
        comm::sync::broadcast::receive_from(rank, pcomm.ref().subCommunicator(rc_comm), tile);
        return ConstTile_t(std::move(tile));
      },
      "recv_tile");

  return detail::handle_recv_tile<D>::call(
      hpx::dataflow(ex, std::move(recv_bcast_f), mpi_task_chain()));
}
}
}
