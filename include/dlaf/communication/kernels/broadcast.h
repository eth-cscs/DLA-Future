//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <mpi.h>

#include <pika/execution.hpp>
#include <pika/future.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/common/callable_object.h"
#include "dlaf/common/data.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"
#include "dlaf/communication/rdma.h"
#include "dlaf/communication/with_contiguous_buffer.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/transform_mpi.h"

namespace dlaf {
namespace comm {

template <class T, Device D>
void sendBcast(const matrix::Tile<const T, D>& tile, common::PromiseGuard<Communicator> pcomm,
               MPI_Request* req) {
#if !defined(DLAF_WITH_CUDA_RDMA)
  static_assert(D == Device::CPU, "DLAF_WITH_CUDA_RDMA=off, MPI accepts just CPU memory.");
#endif

  const auto& comm = pcomm.ref();
  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(
      MPI_Ibcast(const_cast<T*>(msg.data()), msg.count(), msg.mpi_type(), comm.rank(), comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(sendBcast);

template <class T, Device D>
void recvBcast(const matrix::Tile<T, D>& tile, comm::IndexT_MPI root_rank,
               common::PromiseGuard<Communicator> pcomm, MPI_Request* req) {
#if !defined(DLAF_WITH_CUDA_RDMA)
  static_assert(D == Device::CPU, "DLAF_WITH_CUDA_RDMA=off, MPI accepts just CPU memory.");
#endif

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(MPI_Ibcast(msg.data(), msg.count(), msg.mpi_type(), root_rank, pcomm.ref(), req));
}

DLAF_MAKE_CALLABLE_OBJECT(recvBcast);

template <class T, Device D, template <class> class Future, class CommSender>
void scheduleSendBcast(Future<matrix::Tile<const T, D>> tile, CommSender&& pcomm) {
  namespace ex = pika::execution::experimental;

  using dlaf::comm::internal::withCommTile;
  using dlaf::internal::keepIfSharedFuture;
  using dlaf::internal::whenAllLift;

  auto send = [pcomm = std::forward<CommSender>(pcomm)](auto const&, auto const& tile_comm) mutable {
    return whenAllLift(std::cref(tile_comm), std::move(pcomm)) | internal::transformMPI(sendBcast_o);
  };
  ex::start_detached(withCommTile(keepIfSharedFuture(std::move(tile)), std::move(send)));
}

template <class T, Device D, class CommSender>
void scheduleRecvBcast(pika::future<matrix::Tile<T, D>> tile, comm::IndexT_MPI root_rank,
                       CommSender&& pcomm) {
  namespace ex = pika::execution::experimental;

  using dlaf::comm::internal::copyBack;
  using dlaf::comm::internal::transformMPI;
  using dlaf::comm::internal::withSimilarCommTile;
  using dlaf::internal::whenAllLift;

  auto recv_copy_back = [root_rank, pcomm = std::forward<CommSender>(
                                        pcomm)](auto const& tile_in, auto const& tile_comm) mutable {
    auto recv_sender =
        whenAllLift(std::cref(tile_comm), root_rank, std::move(pcomm)) | transformMPI(recvBcast_o);
    return copyBack(std::move(recv_sender), tile_in, tile_comm);
  };
  ex::start_detached(withSimilarCommTile(std::move(tile), std::move(recv_copy_back)));
}

}
}
