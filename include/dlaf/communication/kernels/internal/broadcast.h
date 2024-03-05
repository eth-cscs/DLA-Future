//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <complex>
#include <utility>

#include <mpi.h>

#include <pika/execution.hpp>

#include <dlaf/common/assert.h>
#include <dlaf/common/callable_object.h>
#include <dlaf/common/data.h>
#include <dlaf/common/eti.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/index.h>
#include <dlaf/communication/message.h>
#include <dlaf/communication/rdma.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/sender/traits.h>
#include <dlaf/sender/transform_mpi.h>
#include <dlaf/sender/with_temporary_tile.h>

namespace dlaf::comm::internal {
template <class T, Device D>
void sendBcast(const Communicator& comm, const matrix::Tile<const T, D>& tile, MPI_Request* req) {
#if !defined(DLAF_WITH_MPI_GPU_SUPPORT)
  static_assert(D == Device::CPU, "DLAF_WITH_MPI_GPU_SUPPORT=off, MPI accepts only CPU memory.");
#endif

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(MPI_Ibcast(const_cast<T*>(msg.data()), msg.count(), msg.mpi_type(), comm.rank(),
                                  comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(sendBcast);

template <Device D_comm, dlaf::internal::RequireContiguous require_contiguous, class T, Device D,
          class Comm>
[[nodiscard]] auto scheduleSendBcast(pika::execution::experimental::unique_any_sender<Comm> pcomm,
                                     dlaf::matrix::ReadOnlyTileSender<T, D> tile) {
  using dlaf::internal::CopyFromDestination;
  using dlaf::internal::CopyToDestination;
  using dlaf::internal::SenderSingleValueType;
  using dlaf::internal::whenAllLift;
  using dlaf::internal::withTemporaryTile;

  auto send = [pcomm = std::move(pcomm)](const auto& tile_comm) mutable {
    return whenAllLift(std::move(pcomm), std::cref(tile_comm)) | transformMPI(sendBcast_o);
  };

#if !defined(DLAF_WITH_MPI_GPU_SUPPORT)
  static_assert(D == Device::CPU, "DLAF_WITH_MPI_GPU_SUPPORT=off, MPI accepts only CPU memory.");
#endif

  // The input tile must be copied to the temporary tile used for the send, but
  // the temporary tile does not need to be copied back to the input since the
  // data is not changed by the send. A send does not require contiguous memory.
  return withTemporaryTile<D_comm, CopyToDestination::Yes, CopyFromDestination::No, require_contiguous>(
      std::move(tile), std::move(send));
}

template <class T, Device D>
void recvBcast(const Communicator& comm, comm::IndexT_MPI root_rank, const matrix::Tile<T, D>& tile,
               MPI_Request* req) {
#if !defined(DLAF_WITH_MPI_GPU_SUPPORT)
  static_assert(D == Device::CPU, "DLAF_WITH_MPI_GPU_SUPPORT=off, MPI accepts only CPU memory.");
#endif

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(MPI_Ibcast(msg.data(), msg.count(), msg.mpi_type(), root_rank, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(recvBcast);

template <Device D_comm, dlaf::internal::RequireContiguous require_contiguous, class T, Device D,
          class Comm>
[[nodiscard]] dlaf::matrix::ReadWriteTileSender<T, D> scheduleRecvBcast(
    pika::execution::experimental::unique_any_sender<Comm> pcomm, comm::IndexT_MPI root_rank,
    dlaf::matrix::ReadWriteTileSender<T, D> tile) {
  using dlaf::comm::internal::recvBcast_o;
  using dlaf::internal::CopyFromDestination;
  using dlaf::internal::CopyToDestination;
  using dlaf::internal::SenderSingleValueType;
  using dlaf::internal::whenAllLift;
  using dlaf::internal::withTemporaryTile;

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
  auto recv = [root_rank, pcomm = std::move(pcomm)](const auto& tile_comm) mutable {
    return whenAllLift(std::move(pcomm), root_rank, std::cref(tile_comm)) | transformMPI(recvBcast_o);
  };
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#if !defined(DLAF_WITH_MPI_GPU_SUPPORT)
  static_assert(D == Device::CPU, "DLAF_WITH_MPI_GPU_SUPPORT=off, MPI accepts only CPU memory.");
#endif

  // Since this is a receive we don't need to copy the input to the temporary
  // tile (the input tile may be uninitialized). The received data is copied
  // back from the temporary tile to the input. A receive does not require
  // contiguous memory.
  return withTemporaryTile<D_comm, CopyToDestination::No, CopyFromDestination::Yes, require_contiguous>(
      std::move(tile), std::move(recv));
}
}
