//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <complex>
#include <utility>

#include <mpi.h>

#include <dlaf/common/callable_object.h>
#include <dlaf/common/eti.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/communication/index.h>
#include <dlaf/communication/kernels/p2p.h>
#include <dlaf/communication/message.h>
#include <dlaf/communication/rdma.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/sender/traits.h>
#include <dlaf/sender/transform_mpi.h>
#include <dlaf/sender/when_all_lift.h>
#include <dlaf/sender/with_temporary_tile.h>

namespace dlaf::comm {
namespace internal {
// Non-blocking point to point send
template <class T, Device D>
void send(const Communicator& comm, IndexT_MPI dest, IndexT_MPI tag,
          const matrix::Tile<const T, D>& tile, MPI_Request* req) {
#if !defined(DLAF_WITH_CUDA_RDMA)
  static_assert(D == Device::CPU, "DLAF_WITH_CUDA_RDMA=off, MPI accepts just CPU memory.");
#endif

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(MPI_Isend(msg.data(), msg.count(), msg.mpi_type(), dest, tag, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(send);

template <class CommSender, class Sender>
[[nodiscard]] auto scheduleSend(CommSender&& pcomm, IndexT_MPI dest, IndexT_MPI tag, Sender&& tile) {
  using dlaf::comm::internal::send_o;
  using dlaf::comm::internal::transformMPI;
  using dlaf::internal::CopyFromDestination;
  using dlaf::internal::CopyToDestination;
  using dlaf::internal::RequireContiguous;
  using dlaf::internal::SenderSingleValueType;
  using dlaf::internal::whenAllLift;
  using dlaf::internal::withTemporaryTile;

  auto send = [dest, tag, pcomm = std::forward<CommSender>(pcomm)](const auto& tile_comm) mutable {
    return whenAllLift(std::move(pcomm), dest, tag, std::cref(tile_comm)) | transformMPI(send_o);
  };

  constexpr Device in_device_type = SenderSingleValueType<std::decay_t<Sender>>::device;
  constexpr Device comm_device_type = CommunicationDevice_v<in_device_type>;

  return withTemporaryTile<comm_device_type, CopyToDestination::Yes, CopyFromDestination::No,
                           RequireContiguous::No>(std::forward<Sender>(tile), std::move(send));
}

// Non-blocking point to point receive
template <class T, Device D>
auto recv(const Communicator& comm, IndexT_MPI source, IndexT_MPI tag, const matrix::Tile<T, D>& tile,
          MPI_Request* req) {
#if !defined(DLAF_WITH_CUDA_RDMA)
  static_assert(D == Device::CPU, "DLAF_WITH_CUDA_RDMA=off, MPI accepts just CPU memory.");
#endif

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(MPI_Irecv(msg.data(), msg.count(), msg.mpi_type(), source, tag, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(recv);
}

template <class T, Device D, class CommSender>
[[nodiscard]] pika::execution::experimental::unique_any_sender<> scheduleSend(
    CommSender pcomm, IndexT_MPI dest, IndexT_MPI tag, dlaf::matrix::ReadOnlyTileSender<T, D> tile) {
  return internal::scheduleSend(std::move(pcomm), dest, tag, std::move(tile));
}

// clang-format off
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_SEND_ETI, , pika::execution::experimental::unique_any_sender<Communicator>);
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_SEND_ETI, , pika::execution::experimental::any_sender<Communicator>);
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_SEND_ETI, , CommunicatorPipelineSharedSender);
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_SEND_ETI, , CommunicatorPipelineExclusiveSender);
// clang-format on

template <class T, Device D, class CommSender>
[[nodiscard]] dlaf::matrix::ReadWriteTileSender<T, D> scheduleRecv(
    CommSender pcomm, IndexT_MPI source, IndexT_MPI tag, dlaf::matrix::ReadWriteTileSender<T, D> tile) {
  using dlaf::comm::internal::recv_o;
  using dlaf::comm::internal::transformMPI;
  using dlaf::internal::CopyFromDestination;
  using dlaf::internal::CopyToDestination;
  using dlaf::internal::RequireContiguous;
  using dlaf::internal::SenderSingleValueType;
  using dlaf::internal::whenAllLift;
  using dlaf::internal::withTemporaryTile;

  auto recv = [source, tag, pcomm = std::move(pcomm)](const auto& tile_comm) mutable {
    return whenAllLift(std::move(pcomm), source, tag, std::cref(tile_comm)) | transformMPI(recv_o);
  };

  constexpr Device in_device_type = D;
  constexpr Device comm_device_type = CommunicationDevice_v<in_device_type>;

  return withTemporaryTile<comm_device_type, CopyToDestination::No, CopyFromDestination::Yes,
                           RequireContiguous::No>(std::move(tile), std::move(recv));
}

// clang-format off
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_RECV_ETI, , pika::execution::experimental::unique_any_sender<Communicator>);
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_RECV_ETI, , pika::execution::experimental::any_sender<Communicator>);
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_RECV_ETI, , CommunicatorPipelineSharedSender);
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_RECV_ETI, , CommunicatorPipelineExclusiveSender);
// clang-format on
}
