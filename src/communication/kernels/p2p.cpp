//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <complex>
#include <utility>

#include <mpi.h>

#include "dlaf/common/callable_object.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/kernels/p2p.h"
#include "dlaf/communication/message.h"
#include "dlaf/communication/rdma.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/traits.h"
#include "dlaf/sender/transform_mpi.h"
#include "dlaf/sender/when_all_lift.h"
#include "dlaf/sender/with_temporary_tile.h"

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
  DLAF_MPI_CHECK_ERROR(
      MPI_Isend(const_cast<T*>(msg.data()), msg.count(), msg.mpi_type(), dest, tag, comm, req));
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

  auto recv = [dest, tag, pcomm = std::forward<CommSender>(pcomm)](auto const& tile_comm) mutable {
    return whenAllLift(std::move(pcomm), dest, tag, std::cref(tile_comm)) | transformMPI(send_o);
  };

  constexpr Device in_device_type = SenderSingleValueType<std::decay_t<Sender>>::device;
  constexpr Device comm_device_type = CommunicationDevice_v<in_device_type>;

  return withTemporaryTile<comm_device_type, CopyToDestination::Yes, CopyFromDestination::No,
                           RequireContiguous::No>(std::forward<Sender>(tile), std::move(recv));
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

template <class T, Device D, class Comm>
[[nodiscard]] pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> scheduleSend(
    pika::execution::experimental::unique_any_sender<Comm> pcomm, IndexT_MPI dest, IndexT_MPI tag,
    pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> tile) {
  return internal::scheduleSend(std::move(pcomm), dest, tag, std::move(tile));
}

DLAF_SCHEDULE_SEND_ETI(, float, Device::CPU, Communicator);
DLAF_SCHEDULE_SEND_ETI(, double, Device::CPU, Communicator);
DLAF_SCHEDULE_SEND_ETI(, std::complex<float>, Device::CPU, Communicator);
DLAF_SCHEDULE_SEND_ETI(, std::complex<double>, Device::CPU, Communicator);

DLAF_SCHEDULE_SEND_ETI(, float, Device::CPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_ETI(, double, Device::CPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_ETI(, std::complex<float>, Device::CPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_ETI(, std::complex<double>, Device::CPU, common::PromiseGuard<Communicator>);

#ifdef DLAF_WITH_GPU
DLAF_SCHEDULE_SEND_ETI(, float, Device::GPU, Communicator);
DLAF_SCHEDULE_SEND_ETI(, double, Device::GPU, Communicator);
DLAF_SCHEDULE_SEND_ETI(, std::complex<float>, Device::GPU, Communicator);
DLAF_SCHEDULE_SEND_ETI(, std::complex<double>, Device::GPU, Communicator);

DLAF_SCHEDULE_SEND_ETI(, float, Device::GPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_ETI(, double, Device::GPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_ETI(, std::complex<float>, Device::GPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_ETI(, std::complex<double>, Device::GPU, common::PromiseGuard<Communicator>);
#endif

template <class T, Device D, class Comm>
[[nodiscard]] pika::execution::experimental::unique_any_sender<> scheduleSend(
    pika::execution::experimental::unique_any_sender<Comm> pcomm, IndexT_MPI dest, IndexT_MPI tag,
    pika::execution::experimental::unique_any_sender<pika::shared_future<matrix::Tile<const T, D>>> tile) {
  return internal::scheduleSend(std::move(pcomm), dest, tag, std::move(tile));
}

DLAF_SCHEDULE_SEND_SFTILE_ETI(, float, Device::CPU, Communicator);
DLAF_SCHEDULE_SEND_SFTILE_ETI(, double, Device::CPU, Communicator);
DLAF_SCHEDULE_SEND_SFTILE_ETI(, std::complex<float>, Device::CPU, Communicator);
DLAF_SCHEDULE_SEND_SFTILE_ETI(, std::complex<double>, Device::CPU, Communicator);

DLAF_SCHEDULE_SEND_SFTILE_ETI(, float, Device::CPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_SFTILE_ETI(, double, Device::CPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_SFTILE_ETI(, std::complex<float>, Device::CPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_SFTILE_ETI(, std::complex<double>, Device::CPU, common::PromiseGuard<Communicator>);

#ifdef DLAF_WITH_GPU
DLAF_SCHEDULE_SEND_SFTILE_ETI(, float, Device::GPU, Communicator);
DLAF_SCHEDULE_SEND_SFTILE_ETI(, double, Device::GPU, Communicator);
DLAF_SCHEDULE_SEND_SFTILE_ETI(, std::complex<float>, Device::GPU, Communicator);
DLAF_SCHEDULE_SEND_SFTILE_ETI(, std::complex<double>, Device::GPU, Communicator);

DLAF_SCHEDULE_SEND_SFTILE_ETI(, float, Device::GPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_SFTILE_ETI(, double, Device::GPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_SFTILE_ETI(, std::complex<float>, Device::GPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_SFTILE_ETI(, std::complex<double>, Device::GPU, common::PromiseGuard<Communicator>);
#endif

template <class T, Device D, class Comm>
[[nodiscard]] pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> scheduleRecv(
    pika::execution::experimental::unique_any_sender<Comm> pcomm, IndexT_MPI source, IndexT_MPI tag,
    pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> tile) {
  using dlaf::comm::internal::recv_o;
  using dlaf::comm::internal::transformMPI;
  using dlaf::internal::CopyFromDestination;
  using dlaf::internal::CopyToDestination;
  using dlaf::internal::RequireContiguous;
  using dlaf::internal::SenderSingleValueType;
  using dlaf::internal::whenAllLift;
  using dlaf::internal::withTemporaryTile;

  auto recv = [source, tag, pcomm = std::move(pcomm)](auto const& tile_comm) mutable {
    return whenAllLift(std::move(pcomm), source, tag, std::cref(tile_comm)) | transformMPI(recv_o);
  };

  constexpr Device in_device_type = D;
  constexpr Device comm_device_type = CommunicationDevice_v<in_device_type>;

  return withTemporaryTile<comm_device_type, CopyToDestination::No, CopyFromDestination::Yes,
                           RequireContiguous::No>(std::move(tile), std::move(recv));
}

DLAF_SCHEDULE_RECV_ETI(, float, Device::CPU, Communicator);
DLAF_SCHEDULE_RECV_ETI(, double, Device::CPU, Communicator);
DLAF_SCHEDULE_RECV_ETI(, std::complex<float>, Device::CPU, Communicator);
DLAF_SCHEDULE_RECV_ETI(, std::complex<double>, Device::CPU, Communicator);

DLAF_SCHEDULE_RECV_ETI(, float, Device::CPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_RECV_ETI(, double, Device::CPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_RECV_ETI(, std::complex<float>, Device::CPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_RECV_ETI(, std::complex<double>, Device::CPU, common::PromiseGuard<Communicator>);

#ifdef DLAF_WITH_GPU
DLAF_SCHEDULE_RECV_ETI(, float, Device::GPU, Communicator);
DLAF_SCHEDULE_RECV_ETI(, double, Device::GPU, Communicator);
DLAF_SCHEDULE_RECV_ETI(, std::complex<float>, Device::GPU, Communicator);
DLAF_SCHEDULE_RECV_ETI(, std::complex<double>, Device::GPU, Communicator);

DLAF_SCHEDULE_RECV_ETI(, float, Device::GPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_RECV_ETI(, double, Device::GPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_RECV_ETI(, std::complex<float>, Device::GPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_RECV_ETI(, std::complex<double>, Device::GPU, common::PromiseGuard<Communicator>);
#endif
}
