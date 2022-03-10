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

#include <pika/mpi.hpp>
#include <pika/unwrap.hpp>

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
               const common::PromiseGuard<Communicator>& pcomm, MPI_Request* req) {
#if !defined(DLAF_WITH_CUDA_RDMA)
  static_assert(D == Device::CPU, "DLAF_WITH_CUDA_RDMA=off, MPI accepts just CPU memory.");
#endif

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(MPI_Ibcast(msg.data(), msg.count(), msg.mpi_type(), root_rank, pcomm.ref(), req));
}

DLAF_MAKE_CALLABLE_OBJECT(recvBcast);

template <class T, Device D, template <class> class Future>
void scheduleSendBcast(const comm::Executor& ex, Future<matrix::Tile<const T, D>> tile,
                       pika::future<common::PromiseGuard<Communicator>> pcomm) {
  using dlaf::internal::keepIfSharedFuture;
  using dlaf::internal::whenAllLift;
  using internal::prepareSendTile;
  using matrix::unwrapExtendTiles;
  using pika::execution::experimental::start_detached;
  using pika::mpi::experimental::transform_mpi;

  whenAllLift(keepIfSharedFuture(prepareSendTile(std::move(tile))), std::move(pcomm)) |
      transform_mpi(pika::unwrapping(sendBcast_o)) | start_detached();
}

namespace internal {

template <class T>
struct ScheduleRecvBcast {
  template <Device D>
  static auto call(const comm::Executor& ex, pika::future<matrix::Tile<T, D>> tile,
                   comm::IndexT_MPI root_rank, pika::future<common::PromiseGuard<Communicator>> pcomm) {
#if !defined(DLAF_WITH_CUDA_RDMA)
    static_assert(D == Device::CPU, "With CUDA RDMA disabled, MPI accepts just CPU memory.");
#endif

    using dlaf::internal::whenAllLift;
    using matrix::unwrapExtendTiles;
    using pika::dataflow;
    using pika::execution::experimental::make_future;
    using pika::mpi::experimental::transform_mpi;

    return whenAllLift(std::move(tile), root_rank, std::move(pcomm)) | transform_mpi(recvBcast_o) |
           make_future();
  }

#if defined(DLAF_WITH_CUDA) && !defined(DLAF_WITH_CUDA_RDMA)
  static void call(const comm::Executor& ex, pika::future<matrix::Tile<T, Device::GPU>> tile,
                   comm::IndexT_MPI root_rank, pika::future<common::PromiseGuard<Communicator>> pcomm) {
    using matrix::duplicateIfNeeded;

    // Note:
    //
    // TILE_GPU -+-> duplicateIfNeeded<CPU> ---> TILE_CPU ---> recvBcast ---> TILE_CPU -+-> copy
    //           |                                                                      |
    //           +----------------------------------------------------------------------+
    //
    // Actually `duplicateIfNeeded` always makes a copy, because it is always needed since this
    // is the specialization for GPU input and MPI without CUDA_RDMA requires CPU memory.

    auto tile_gpu = tile.share();
    auto tile_cpu = duplicateIfNeeded<Device::CPU>(tile_gpu);

    tile_cpu = std::move(pika::get<0>(pika::split_future(
        ScheduleRecvBcast<T>::call(ex, std::move(tile_cpu), root_rank, std::move(pcomm)))));

    pika::execution::experimental::when_all(dlaf::internal::keepIfSharedFuture(std::move(tile_cpu)),
                                            dlaf::internal::keepIfSharedFuture(std::move(tile_gpu))) |
        dlaf::matrix::copy(
            dlaf::internal::Policy<dlaf::matrix::internal::CopyBackend_v<Device::CPU, Device::GPU>>()) |
        pika::execution::experimental::start_detached();
  }
#endif
};

}

template <class T, Device D>
void scheduleRecvBcast(const comm::Executor& ex, pika::future<matrix::Tile<T, D>> tile,
                       comm::IndexT_MPI root_rank,
                       pika::future<common::PromiseGuard<Communicator>> pcomm) {
  internal::ScheduleRecvBcast<T>::call(ex, std::move(tile), root_rank, std::move(pcomm));
}

}
}
