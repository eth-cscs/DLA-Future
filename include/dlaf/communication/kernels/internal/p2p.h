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

/// @file

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
// Non-blocking point to point send
template <class T, Device D>
void send(const Communicator& comm, IndexT_MPI dest, IndexT_MPI tag,
          const matrix::Tile<const T, D>& tile, MPI_Request* req) {
#if !defined(DLAF_WITH_MPI_GPU_AWARE)
  static_assert(D == Device::CPU, "DLAF_WITH_MPI_GPU_AWARE=off, MPI accepts only CPU memory.");
#endif

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(MPI_Isend(msg.data(), msg.count(), msg.mpi_type(), dest, tag, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(send);

template <Device D_comm, dlaf::internal::RequireContiguous require_contiguous, class T, Device D,
          class CommSender>
[[nodiscard]] auto schedule_send(CommSender pcomm, IndexT_MPI dest, IndexT_MPI tag,
                                 dlaf::matrix::ReadOnlyTileSender<T, D> tile) {
  using dlaf::comm::internal::send_o;
  using dlaf::internal::CopyFromDestination;
  using dlaf::internal::CopyToDestination;
  using dlaf::internal::SenderSingleValueType;
  using dlaf::internal::whenAllLift;
  using dlaf::internal::withTemporaryTile;

  auto send = [dest, tag, pcomm = std::move(pcomm)](const auto& tile_comm) mutable {
    return whenAllLift(std::move(pcomm), dest, tag, std::cref(tile_comm)) | transformMPI(send_o);
  };

#if !defined(DLAF_WITH_MPI_GPU_AWARE)
  static_assert(D_comm == Device::CPU, "DLAF_WITH_MPI_GPU_AWARE=off, MPI accepts only CPU memory.");
#endif

  return withTemporaryTile<D_comm, CopyToDestination::Yes, CopyFromDestination::No, require_contiguous>(
      std::move(tile), std::move(send));
}

// Non-blocking point to point receive
template <class T, Device D>
auto recv(const Communicator& comm, IndexT_MPI source, IndexT_MPI tag, const matrix::Tile<T, D>& tile,
          MPI_Request* req) {
#if !defined(DLAF_WITH_MPI_GPU_AWARE)
  static_assert(D == Device::CPU, "DLAF_WITH_MPI_GPU_AWARE=off, MPI accepts only CPU memory.");
#endif

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(MPI_Irecv(msg.data(), msg.count(), msg.mpi_type(), source, tag, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(recv);

template <Device D_comm, dlaf::internal::RequireContiguous require_contiguous, class T, Device D,
          class CommSender>
[[nodiscard]] dlaf::matrix::ReadWriteTileSender<T, D> schedule_recv(
    CommSender pcomm, IndexT_MPI source, IndexT_MPI tag, dlaf::matrix::ReadWriteTileSender<T, D> tile) {
  using dlaf::comm::internal::recv_o;
  using dlaf::internal::CopyFromDestination;
  using dlaf::internal::CopyToDestination;
  using dlaf::internal::SenderSingleValueType;
  using dlaf::internal::whenAllLift;
  using dlaf::internal::withTemporaryTile;

  auto recv = [source, tag, pcomm = std::move(pcomm)](const auto& tile_comm) mutable {
    return whenAllLift(std::move(pcomm), source, tag, std::cref(tile_comm)) | transformMPI(recv_o);
  };

#if !defined(DLAF_WITH_MPI_GPU_AWARE)
  static_assert(D_comm == Device::CPU, "DLAF_WITH_MPI_GPU_AWARE=off, MPI accepts only CPU memory.");
#endif

  return withTemporaryTile<D_comm, CopyToDestination::No, CopyFromDestination::Yes, require_contiguous>(
      std::move(tile), std::move(recv));
}

}
