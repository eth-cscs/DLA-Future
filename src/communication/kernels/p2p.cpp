//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
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
#include <dlaf/communication/kernels/internal/p2p.h>
#include <dlaf/communication/kernels/p2p.h>
#include <dlaf/communication/message.h>
#include <dlaf/communication/rdma.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/sender/traits.h>
#include <dlaf/sender/transform_mpi.h>
#include <dlaf/sender/when_all_lift.h>
#include <dlaf/sender/with_temporary_tile.h>

namespace dlaf::comm {

template <class T, Device D, class CommSender>
[[nodiscard]] pika::execution::experimental::unique_any_sender<> schedule_send(
    CommSender pcomm, IndexT_MPI dest, IndexT_MPI tag, dlaf::matrix::ReadOnlyTileSender<T, D> tile) {
  using dlaf::internal::RequireContiguous;
  constexpr Device D_comm = CommunicationDevice_v<D>;
  constexpr auto require_contiguous =
#if defined(DLAF_WITH_MPI_GPU_SUPPORT) && defined(DLAF_WITH_MPI_GPU_SUPPORT_FORCE_CONTIGUOUS)
      D_comm == Device::GPU ? RequireContiguous::Yes :
#endif
                            RequireContiguous::No;
  return internal::schedule_send<D_comm, require_contiguous>(std::move(pcomm), dest, tag,
                                                             std::move(tile));
}

// clang-format off
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_SEND_ETI, , pika::execution::experimental::unique_any_sender<Communicator>);
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_SEND_ETI, , pika::execution::experimental::any_sender<Communicator>);
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_SEND_ETI, , CommunicatorPipelineSharedSender);
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_SEND_ETI, , CommunicatorPipelineExclusiveSender);
// clang-format on

template <class T, Device D, class CommSender>
[[nodiscard]] dlaf::matrix::ReadWriteTileSender<T, D> schedule_recv(
    CommSender pcomm, IndexT_MPI source, IndexT_MPI tag, dlaf::matrix::ReadWriteTileSender<T, D> tile) {
  using dlaf::internal::RequireContiguous;
  constexpr Device D_comm = CommunicationDevice_v<D>;
  constexpr auto require_contiguous =
#if defined(DLAF_WITH_MPI_GPU_SUPPORT) && defined(DLAF_WITH_MPI_GPU_SUPPORT_FORCE_CONTIGUOUS)
      D_comm == Device::GPU ? RequireContiguous::Yes :
#endif
                            RequireContiguous::No;

  return internal::schedule_recv<D_comm, require_contiguous>(std::move(pcomm), source, tag,
                                                             std::move(tile));
}

// clang-format off
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_RECV_ETI, , pika::execution::experimental::unique_any_sender<Communicator>);
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_RECV_ETI, , pika::execution::experimental::any_sender<Communicator>);
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_RECV_ETI, , CommunicatorPipelineSharedSender);
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_RECV_ETI, , CommunicatorPipelineExclusiveSender);
// clang-format on
}
