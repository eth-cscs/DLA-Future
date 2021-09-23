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

#include <hpx/local/unwrap.hpp>

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

template <class T, Device D>
void sendBcast(const matrix::Tile<const T, D>& tile, const common::PromiseGuard<Communicator>& pcomm,
               MPI_Request* req) {
  const auto& comm = pcomm.ref();
  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CALL(
      MPI_Ibcast(const_cast<T*>(msg.data()), msg.count(), msg.mpi_type(), comm.rank(), comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(sendBcast);

template <class T, Device D>
void recvBcast(const matrix::Tile<T, D>& tile, comm::IndexT_MPI root_rank,
               const common::PromiseGuard<Communicator>& pcomm, MPI_Request* req) {
  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CALL(MPI_Ibcast(msg.data(), msg.count(), msg.mpi_type(), root_rank, pcomm.ref(), req));
}

DLAF_MAKE_CALLABLE_OBJECT(recvBcast);

template <class T, Device D, template <class> class Future>
void scheduleSendBcast(const comm::Executor& ex, Future<matrix::Tile<const T, D>> tile,
                       hpx::future<common::PromiseGuard<Communicator>> pcomm) {
  using matrix::unwrapExtendTiles;
  using internal::prepareSendTile;
  hpx::dataflow(ex, unwrapExtendTiles(sendBcast_o), prepareSendTile(std::move(tile)), std::move(pcomm));
}

template <class T, Device D>
void scheduleRecvBcast(const comm::Executor& ex, hpx::future<matrix::Tile<T, D>> tile,
                       comm::IndexT_MPI root_rank,
                       hpx::future<common::PromiseGuard<Communicator>> pcomm) {
  using matrix::unwrapExtendTiles;
  hpx::dataflow(ex, unwrapExtendTiles(recvBcast_o), std::move(tile), root_rank, std::move(pcomm));
}

}
}
