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

#include "dlaf/common/assert.h"
#include "dlaf/common/callable_object.h"
#include "dlaf/common/data.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"
#include "dlaf/communication/rdma.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/transform_mpi.h"
#include "dlaf/sender/with_temporary_tile.h"

namespace dlaf::comm {
namespace internal {
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
}

template <class TileSender, class CommSender>
auto scheduleSendBcast(TileSender&& tile, CommSender&& pcomm) {
  using dlaf::comm::internal::CopyFromDestination;
  using dlaf::comm::internal::CopyToDestination;
  using dlaf::comm::internal::RequireContiguous;
  using dlaf::comm::internal::sendBcast_o;
  using dlaf::comm::internal::transformMPI;
  using dlaf::comm::internal::withTemporaryTile;
  using dlaf::internal::SenderSingleValueType;
  using dlaf::internal::whenAllLift;

  auto send = [pcomm = std::forward<CommSender>(pcomm)](auto const& tile_comm) mutable {
    return whenAllLift(std::cref(tile_comm), std::move(pcomm)) | transformMPI(sendBcast_o);
  };

  constexpr Device in_device_type = SenderSingleValueType<std::decay_t<TileSender>>::D;
  constexpr Device comm_device_type = CommunicationDevice_v<in_device_type>;

  return withTemporaryTile<comm_device_type, CopyToDestination::Yes, CopyFromDestination::No,
                           RequireContiguous::No>(std::forward<TileSender>(tile), std::move(send));
}

template <class TileSender, class CommSender>
auto scheduleRecvBcast(TileSender&& tile, comm::IndexT_MPI root_rank, CommSender&& pcomm) {
  using dlaf::comm::internal::CopyFromDestination;
  using dlaf::comm::internal::CopyToDestination;
  using dlaf::comm::internal::recvBcast_o;
  using dlaf::comm::internal::RequireContiguous;
  using dlaf::comm::internal::transformMPI;
  using dlaf::comm::internal::withTemporaryTile;
  using dlaf::internal::SenderSingleValueType;
  using dlaf::internal::whenAllLift;

  auto recv = [root_rank, pcomm = std::forward<CommSender>(pcomm)](auto const& tile_comm) mutable {
    return whenAllLift(std::cref(tile_comm), root_rank, std::move(pcomm)) | transformMPI(recvBcast_o);
  };

  constexpr Device in_device_type = SenderSingleValueType<std::decay_t<TileSender>>::D;
  constexpr Device comm_device_type = CommunicationDevice_v<in_device_type>;

  return withTemporaryTile<comm_device_type, CopyToDestination::No, CopyFromDestination::Yes,
                           RequireContiguous::No>(std::forward<TileSender>(tile), std::move(recv));
}
}
