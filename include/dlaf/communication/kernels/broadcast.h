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
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/transform_mpi.h"

namespace dlaf {
namespace comm {

template <class T, Device D>
void sendBcast(const matrix::Tile<const T, D>& tile, const Communicator& comm, MPI_Request* req) {
#if !defined(DLAF_WITH_CUDA_RDMA)
  static_assert(D == Device::CPU, "DLAF_WITH_CUDA_RDMA=off, MPI accepts just CPU memory.");
#endif

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(
      MPI_Ibcast(const_cast<T*>(msg.data()), msg.count(), msg.mpi_type(), comm.rank(), comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(sendBcast);

template <class T, Device D>
void recvBcast(const matrix::Tile<T, D>& tile, comm::IndexT_MPI root_rank, const Communicator& comm,
               MPI_Request* req) {
#if !defined(DLAF_WITH_CUDA_RDMA)
  static_assert(D == Device::CPU, "DLAF_WITH_CUDA_RDMA=off, MPI accepts just CPU memory.");
#endif

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(MPI_Ibcast(msg.data(), msg.count(), msg.mpi_type(), root_rank, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(recvBcast);

template <class T, Device D, template <class> class Future, class CommSender>
void scheduleSendBcast(Future<matrix::Tile<const T, D>> tile, CommSender&& pcomm) {
  using dlaf::internal::keepIfSharedFuture;
  using dlaf::internal::whenAllLift;
  using internal::prepareSendTile;
  using pika::execution::experimental::start_detached;

  whenAllLift(keepIfSharedFuture(prepareSendTile(std::move(tile))), std::forward<CommSender>(pcomm)) |
      internal::transformMPI(sendBcast_o) | start_detached();
}

namespace internal {

template <class T>
struct ScheduleRecvBcast {
  template <Device D, class CommSender>
  static auto call(pika::future<matrix::Tile<T, D>> tile, comm::IndexT_MPI root_rank,
                   CommSender&& pcomm) {
#if !defined(DLAF_WITH_CUDA_RDMA)
    static_assert(D == Device::CPU, "With CUDA RDMA disabled, MPI accepts just CPU memory.");
#endif

    using dlaf::internal::whenAllLift;
    using pika::execution::experimental::start_detached;

    return whenAllLift(std::move(tile), root_rank, std::forward<CommSender>(pcomm)) |
           transformMPI(recvBcast_o);
  }

#if defined(DLAF_WITH_CUDA) && !defined(DLAF_WITH_CUDA_RDMA)
  template <class CommSender>
  static auto call(pika::future<matrix::Tile<T, Device::GPU>> tile_gpu, comm::IndexT_MPI root_rank,
                   CommSender&& pcomm) {
    // Note:
    // TILE_GPU -+-> duplicate<CPU> ---> TILE_CPU ---> recvBcast ---> TILE_CPU -+-> copy
    //           |                                                              |
    //           +--------------------------------------------------------------+
    namespace ex = pika::execution::experimental;

    using dlaf::internal::Policy;
    using dlaf::internal::transform;
    using dlaf::internal::whenAllLift;
    using dlaf::matrix::copy;
    using dlaf::matrix::Duplicate;
    using dlaf::matrix::Tile;
    using pika::threads::thread_priority;

    return std::move(tile_gpu) |
           // Start an asynchronous scope for keeping the GPU tile alive until
           // data has been copied back into it.
           ex::let_value([=, pcomm = std::forward<CommSender>(pcomm)](
                             Tile<T, Device::GPU>& tile_gpu) mutable {
             // Create a CPU tile with the same dimensions as the GPU tile.
             return ex::just(std::cref(tile_gpu)) |
                    transform(Policy<Backend::GPU>(thread_priority::high), Duplicate<Device::CPU>{}) |
                    // Start an asynchronous scope for keeping the CPU tile
                    // alive until data has been copied away from it.
                    ex::let_value([=, pcomm = std::move(pcomm),
                                   &tile_gpu](Tile<T, Device::CPU>& tile_cpu) mutable {
                      // Perform the actual receive into the CPU tile.
                      auto recv_sender = whenAllLift(std::cref(tile_cpu), root_rank, std::move(pcomm)) |
                                         transformMPI(recvBcast_o);
                      // Copy the received data from the CPU tile to the
                      // GPU tile.
                      return whenAllLift(std::move(recv_sender), std::cref(tile_cpu),
                                         std::cref(tile_gpu)) |
                             copy(Policy<Backend::GPU>(thread_priority::high));
                    });
           });
  }
#endif
};

}

template <class T, Device D, class CommSender>
void scheduleRecvBcast(pika::future<matrix::Tile<T, D>> tile, comm::IndexT_MPI root_rank,
                       CommSender&& pcomm) {
  internal::ScheduleRecvBcast<T>::call(std::move(tile), root_rank, std::forward<CommSender>(pcomm)) |
      pika::execution::experimental::start_detached();
}

}
}
