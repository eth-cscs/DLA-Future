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
void bcast_send(matrix::Tile<const T, Device::CPU> const& tile, common::PromiseGuard<Communicator> pcomm,
                MPI_Request* req) {
  auto msg = comm::make_message(common::make_data(tile));
  MPI_Ibcast(const_cast<T*>(msg.data()), msg.count(), msg.mpi_type(), pcomm.ref().rank(), pcomm.ref(),
             req);
}

// Non-blocking receiver broadcast
template <class T>
matrix::Tile<const T, Device::CPU> bcast_recv(TileElementSize tile_size, int root_rank,
                                              common::PromiseGuard<Communicator> pcomm,
                                              MPI_Request* req) {
  using Tile_t = matrix::Tile<T, Device::CPU>;
  using ConstTile_t = matrix::Tile<const T, Device::CPU>;
  using MemView_t = memory::MemoryView<T, Device::CPU>;

  MemView_t mem_view(util::size_t::mul(tile_size.rows(), tile_size.cols()));
  Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());

  auto msg = comm::make_message(common::make_data(tile));
  MPI_Ibcast(msg.data(), msg.count(), msg.mpi_type(), root_rank, pcomm.ref(), req);
  return ConstTile_t(std::move(tile));
}

}
}
