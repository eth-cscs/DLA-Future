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

// Non-blocking sender broadcast
template <class T>
static void bcast_send(matrix::Tile<const T, Device::CPU> const& tile,
                       common::PromiseGuard<Communicator> pcomm, MPI_Request* req) {
  auto msg = comm::make_message(common::make_data(tile));
  MPI_Ibcast(const_cast<T*>(msg.data()), msg.count(), msg.mpi_type(), pcomm.ref().rank(), pcomm.ref(),
             req);
}

// Non-blocking receiver broadcast
template <class T>
static auto bcast_recv(TileElementSize tile_size, int root_rank,
                       common::PromiseGuard<Communicator> pcomm, MPI_Request* req) {
  using Tile_t = matrix::Tile<T, Device::CPU>;
  using ConstTile_t = matrix::Tile<const T, Device::CPU>;
  using MemView_t = memory::MemoryView<T, Device::CPU>;

  MemView_t mem_view(util::size_t::mul(tile_size.rows(), tile_size.cols()));
  Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());

  auto msg = comm::make_message(common::make_data(tile));
  MPI_Ibcast(msg.data(), msg.count(), msg.mpi_type(), root_rank, pcomm.ref(), req);
  return ConstTile_t(std::move(tile));
}

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
