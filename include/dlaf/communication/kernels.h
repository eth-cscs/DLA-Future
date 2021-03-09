//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <hpx/include/parallel_executors.hpp>
#include <hpx/modules/threading_base.hpp>

#include <mpi.h>

#include "dlaf/common/assert.h"
#include "dlaf/common/data.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/message.h"
#include "dlaf/matrix/tile.h"

namespace dlaf {
namespace comm {

template <class T, MPIMech M>
struct bcast {
  // Non-blocking sender broadcast
  static void send(matrix::Tile<const T, Device::CPU> const& tile, Communicator comm, MPI_Request* req) {
    auto msg = comm::make_message(common::make_data(tile));
    MPI_Ibcast(const_cast<T*>(msg.data()), msg.count(), msg.mpi_type(), comm.rank(), comm, req);
  }

  // Non-blocking receiver broadcast
  static auto recv(TileElementSize tile_size, int root_rank, Communicator comm, MPI_Request* req) {
    using Tile_t = matrix::Tile<T, Device::CPU>;
    using ConstTile_t = matrix::Tile<const T, Device::CPU>;
    using MemView_t = memory::MemoryView<T, Device::CPU>;

    MemView_t mem_view(util::size_t::mul(tile_size.rows(), tile_size.cols()));
    Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());

    auto msg = comm::make_message(common::make_data(tile));
    MPI_Ibcast(msg.data(), msg.count(), msg.mpi_type(), root_rank, comm, req);
    return ConstTile_t(std::move(tile));
  }
};

template <class T>
struct bcast<T, MPIMech::Blocking> {
  // Blocking sender broadcast
  static void send(matrix::Tile<const T, Device::CPU> const& tile, Communicator comm) {
    auto msg = comm::make_message(common::make_data(tile));
    MPI_Bcast(const_cast<T*>(msg.data()), msg.count(), msg.mpi_type(), comm.rank(), comm);
  }

  // Blocking receiver broadcast
  static auto recv(TileElementSize tile_size, int root_rank, Communicator comm) {
    using Tile_t = matrix::Tile<T, Device::CPU>;
    using ConstTile_t = matrix::Tile<const T, Device::CPU>;
    using MemView_t = memory::MemoryView<T, Device::CPU>;

    MemView_t mem_view(util::size_t::mul(tile_size.rows(), tile_size.cols()));
    Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());

    auto msg = comm::make_message(common::make_data(tile));
    MPI_Bcast(msg.data(), msg.count(), msg.mpi_type(), root_rank, comm);
    return ConstTile_t(std::move(tile));
  }
};

/// MPI_Isend wrapper
///
/// For more information, see the Data concept in "dlaf/common/data.h"
// template <class DataIn, MPIMech M>
// hpx::future<void> send(Executor<M>& ex, int receiver_rank, const DataIn& data) {
//  int tag = 0;
//  auto message = make_message(common::make_data(data));
//  return ex.async_execute(MPI_Isend, message.data(), message.count(), message.mpi_type(), receiver_rank,
//                          tag);
//}

/// MPI_Irecv wrapper
///
/// For more information, see the Data concept in "dlaf/common/data.h"
// template <class DataOut, MPIMech M>
// hpx::future<void> recv(Executor<M>& ex, int sender_rank, DataOut& data) {
//  int tag = 0;
//  auto message = make_message(common::make_data(data));
//  return ex.async_execute(MPI_Irecv, message.data(), message.count(), message.mpi_type(), sender_rank,
//                          tag);
//}

/// MPI_Ibcast wrapper
///
/// For more information, see the Data concept in "dlaf/common/data.h"
// template <class DataIn, MPIMech M>
// hpx::future<void> bcast(Executor<M>& ex, int root_rank, DataIn& tile) {
//  auto data = common::make_data(tile);
//  using DataT = std::remove_const_t<typename common::data_traits<decltype(data)>::element_t>;
//
//  auto message = comm::make_message(common::make_data(data));
//  return ex.async_execute(MPI_Ibcast, const_cast<DataT*>(message.data()), message.count(),
//                          message.mpi_type(), root_rank);
//}

// template <class T, MPIMech M>
// void bcast_send_tile(hpx::execution::parallel_executor executor_hp,
//                     common::Pipeline<Executor<M>>& mpi_task_chain,
//                     hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile) {
//  using ConstTile_t = matrix::Tile<const T, Device::CPU>;
//  using PromiseExec_t = common::PromiseGuard<Executor<M>>;
//
//  // Broadcast the (trailing) panel column-wise
//  auto send_bcast_f = hpx::util::annotated_function(
//      [](hpx::shared_future<ConstTile_t> ftile, hpx::future<PromiseExec_t> fpex) {
//        const ConstTile_t& tile = ftile.get();
//        hpx::future<void> comm_fut;
//        {
//          PromiseExec_t pex = fpex.get();
//          comm_fut = comm::bcast(pex.ref(), pex.ref().comm().rank(), tile);
//        }
//        comm_fut.get();
//      },
//      "send_tile");
//  hpx::dataflow(executor_hp, std::move(send_bcast_f), tile, mpi_task_chain());
//}
//
// template <class T, MPIMech M>
// hpx::future<matrix::Tile<const T, Device::CPU>> bcast_recv_tile(
//    hpx::execution::parallel_executor executor_hp, common::Pipeline<Executor<M>>& mpi_task_chain,
//    TileElementSize tile_size, int rank) {
//  using ConstTile_t = matrix::Tile<const T, Device::CPU>;
//  using PromiseExec_t = common::PromiseGuard<Executor<M>>;
//  using MemView_t = memory::MemoryView<T, Device::CPU>;
//  using Tile_t = matrix::Tile<T, Device::CPU>;
//
//  // Update the (trailing) panel column-wise
//  auto recv_bcast_f = hpx::util::annotated_function(
//      [rank, tile_size](hpx::future<PromiseExec_t> fpex) {
//        MemView_t mem_view(util::size_t::mul(tile_size.rows(), tile_size.cols()));
//        Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());
//        PromiseExec_t pex = fpex.get();
//        return comm::bcast(pex.ref(), rank, tile)
//            .then(hpx::launch::sync, [t = std::move(tile)](hpx::future<void>) mutable -> ConstTile_t {
//              return std::move(t);
//            });
//      },
//      "recv_tile");
//  return hpx::dataflow(executor_hp, std::move(recv_bcast_f), mpi_task_chain());
//}

/// MPI_Ireduce wrapper
///
/// @param reduce_operation MPI_Op to perform on @p input data coming from ranks in @p communicator
// template <class DataIn, class DataOut, MPIMech M>
// hpx::future<void> reduce(Executor<M>& ex, int root_rank, MPI_Op reduce_operation, const DataIn& input,
//                         const DataOut& output) {
//  DLAF_ASSERT(input.is_contiguous(), "Input data has to be contiguous!");
//  if (ex.comm().rank() == root_rank)
//    DLAF_ASSERT(output.is_contiguous(), "Output data has to be contiguous!");
//
//  auto in_msg = comm::make_message(common::make_data(input));
//  auto out_msg = comm::make_message(common::make_data(output));
//  return ex.async_execute(MPI_Ireduce, in_msg.data(), out_msg.data(), in_msg.count(), in_msg.mpi_type(),
//                          reduce_operation, root_rank);
//}

}
}
