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

#include <pika/execution.hpp>

#include <dlaf/common/assert.h>
#include <dlaf/common/callable_object.h>
#include <dlaf/common/data.h>
#include <dlaf/common/eti.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/index.h>
#include <dlaf/communication/kernels/broadcast.h>
#include <dlaf/communication/kernels/internal/broadcast.h>
#include <dlaf/communication/message.h>
#include <dlaf/communication/rdma.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/sender/traits.h>
#include <dlaf/sender/transform_mpi.h>
#include <dlaf/sender/with_temporary_tile.h>

namespace dlaf::comm {

template <class T, Device D, class Comm>
[[nodiscard]] pika::execution::experimental::unique_any_sender<> scheduleSendBcast(
    pika::execution::experimental::unique_any_sender<Comm> pcomm,
    dlaf::matrix::ReadOnlyTileSender<T, D> tile) {
  using dlaf::internal::RequireContiguous;
  constexpr Device D_comm = CommunicationDevice_v<D>;
  constexpr auto require_contiguous =
#if defined(DLAF_WITH_MPI_GPU_SUPPORT) && defined(DLAF_WITH_MPI_GPU_SUPPORT_FORCE_CONTIGUOUS)
      D_comm == Device::GPU ? RequireContiguous::Yes :
#endif
                            RequireContiguous::No;

  return internal::scheduleSendBcast<D_comm, require_contiguous>(std::move(pcomm), std::move(tile));
}

// clang-format off
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_SEND_BCAST_ETI, , CommunicatorPipelineExclusiveWrapper);

DLAF_SCHEDULE_SEND_BCAST_ETI(, SizeType, Device::CPU, CommunicatorPipelineExclusiveWrapper);
#ifdef DLAF_WITH_GPU
DLAF_SCHEDULE_SEND_BCAST_ETI(, SizeType, Device::GPU, CommunicatorPipelineExclusiveWrapper);
#endif
// clang-format on

template <class T, Device D, class Comm>
[[nodiscard]] dlaf::matrix::ReadWriteTileSender<T, D> scheduleRecvBcast(
    pika::execution::experimental::unique_any_sender<Comm> pcomm, comm::IndexT_MPI root_rank,
    dlaf::matrix::ReadWriteTileSender<T, D> tile) {
  using dlaf::internal::RequireContiguous;
  constexpr Device D_comm = CommunicationDevice_v<D>;
  constexpr auto require_contiguous =
#if defined(DLAF_WITH_MPI_GPU_SUPPORT) && defined(DLAF_WITH_MPI_GPU_SUPPORT_FORCE_CONTIGUOUS)
      D_comm == Device::GPU ? RequireContiguous::Yes :
#endif
                            RequireContiguous::No;
  return internal::scheduleRecvBcast<D_comm, require_contiguous>(std::move(pcomm), root_rank,
                                                                 std::move(tile));
}

// clang-format off
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_RECV_BCAST_ETI, , CommunicatorPipelineExclusiveWrapper);

DLAF_SCHEDULE_RECV_BCAST_ETI(, SizeType, Device::CPU, CommunicatorPipelineExclusiveWrapper);
#ifdef DLAF_WITH_GPU
DLAF_SCHEDULE_RECV_BCAST_ETI(, SizeType, Device::GPU, CommunicatorPipelineExclusiveWrapper);
#endif
// clang-format on
}
