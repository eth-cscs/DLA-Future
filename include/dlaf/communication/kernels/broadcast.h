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
void sendBcast(const Communicator& comm, const matrix::Tile<const T, D>& tile, MPI_Request* req) {
#if !defined(DLAF_WITH_CUDA_RDMA)
  static_assert(D == Device::CPU, "DLAF_WITH_CUDA_RDMA=off, MPI accepts just CPU memory.");
#endif

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(
      MPI_Ibcast(const_cast<T*>(msg.data()), msg.count(), msg.mpi_type(), comm.rank(), comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(sendBcast);
}

/// Schedule a broadcast send.
///
/// The returned sender signals completion when the send is done. If the input
/// tile is movable it will be sent by the returned sender. Otherwise a void
/// sender is returned.
template <class TileSender>
[[nodiscard]] auto scheduleSendBcast(
    pika::execution::experimental::unique_any_sender<dlaf::common::PromiseGuard<Communicator>> pcomm,
    TileSender&& tile) {
  using dlaf::comm::internal::sendBcast_o;
  using dlaf::comm::internal::transformMPI;
  using dlaf::internal::CopyFromDestination;
  using dlaf::internal::CopyToDestination;
  using dlaf::internal::RequireContiguous;
  using dlaf::internal::SenderSingleValueType;
  using dlaf::internal::whenAllLift;
  using dlaf::internal::withTemporaryTile;

  auto send = [pcomm = std::move(pcomm)](auto const& tile_comm) mutable {
    return whenAllLift(std::move(pcomm), std::cref(tile_comm)) | transformMPI(sendBcast_o);
  };

  constexpr Device in_device_type = SenderSingleValueType<std::decay_t<TileSender>>::device;
  constexpr Device comm_device_type = CommunicationDevice_v<in_device_type>;

  // The input tile must be copied to the temporary tile used for the send, but
  // the temporary tile does not need to be copied back to the input since the
  // data is not changed by the send. A send does not require contiguous memory.
  return withTemporaryTile<comm_device_type, CopyToDestination::Yes, CopyFromDestination::No,
                           RequireContiguous::No>(std::forward<TileSender>(tile), std::move(send));
}

/// Schedule a broadcast receive.
///
/// The returned sender signals completion when the receive is done. The input
/// sender tile must be writable so that the received data can be written to it.
/// The input tile is sent by the returned sender.
template <class T, Device D>
[[nodiscard]] pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> scheduleRecvBcast(
    pika::execution::experimental::unique_any_sender<dlaf::common::PromiseGuard<Communicator>> pcomm,
    comm::IndexT_MPI root_rank,
    pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> tile);

#define DLAF_SCHEDULE_RECV_BCAST_ETI(kword, Type, Device)                                     \
  kword template pika::execution::experimental::unique_any_sender<matrix::Tile<Type, Device>> \
  scheduleRecvBcast(pika::execution::experimental::unique_any_sender<                         \
                        dlaf::common::PromiseGuard<Communicator>>                             \
                        pcomm,                                                                \
                    comm::IndexT_MPI root_rank,                                               \
                    pika::execution::experimental::unique_any_sender<matrix::Tile<Type, Device>> tile)

DLAF_SCHEDULE_RECV_BCAST_ETI(extern, float, Device::CPU);
DLAF_SCHEDULE_RECV_BCAST_ETI(extern, double, Device::CPU);
DLAF_SCHEDULE_RECV_BCAST_ETI(extern, std::complex<float>, Device::CPU);
DLAF_SCHEDULE_RECV_BCAST_ETI(extern, std::complex<double>, Device::CPU);
}
