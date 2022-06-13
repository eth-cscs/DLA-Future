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
  using dlaf::internal::keepIfSharedFuture;
  using dlaf::internal::whenAllLift;
  using pika::execution::experimental::start_detached;

  auto f = [pcomm = std::forward<CommSender>(pcomm)](auto const&, auto const& tile_comm) mutable {
    return whenAllLift(std::cref(tile_comm), std::move(pcomm)) | internal::transformMPI(sendBcast_o);
  };
  start_detached(internal::with_comm_tile(keepIfSharedFuture(std::move(tile)), std::move(f)));
}

template <class T, Device D, class CommSender>
void scheduleRecvBcast(pika::future<matrix::Tile<T, D>> tile, comm::IndexT_MPI root_rank,
                       CommSender&& pcomm) {
  namespace ex = pika::execution::experimental;

  using dlaf::comm::internal::transformMPI;
  using dlaf::comm::internal::with_similar_comm_tile;
  using dlaf::internal::Policy;
  using dlaf::internal::whenAllLift;
  using dlaf::matrix::copy;
  using pika::execution::experimental::start_detached;
  using pika::threads::thread_priority;

  auto recv_copy_back = [=, pcomm = std::forward<CommSender>(pcomm)](auto const& tile_in,
                                                                     auto const& tile_comm) mutable {
    constexpr Device in_device_type = std::decay_t<decltype(tile_in)>::D;
    constexpr Device comm_device_type = std::decay_t<decltype(tile_comm)>::D;
    constexpr Backend copy_backend =
        dlaf::matrix::internal::CopyBackend_v<in_device_type, comm_device_type>;

    // Perform the receive into a CPU tile
    auto recv_sender =
        whenAllLift(std::cref(tile_comm), root_rank, std::move(pcomm)) | transformMPI(recvBcast_o);

    // This is "copy back if needed". Separate helper? copyIfNeeded?
    if constexpr (in_device_type == comm_device_type) {
      return recv_sender;
    }
    else {
      // Copy the received data from the communication tile to the input tile.
      return whenAllLift(std::move(recv_sender), std::cref(tile_comm), std::cref(tile_in)) |
             copy(Policy<copy_backend>(thread_priority::high));
    }
  };
  start_detached(with_similar_comm_tile(std::move(tile), std::move(recv_copy_back)));
}

}
}
