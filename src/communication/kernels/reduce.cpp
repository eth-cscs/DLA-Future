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

#include <dlaf/common/data.h>
#include <dlaf/common/eti.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/communication/index.h>
#include <dlaf/communication/kernels/internal/reduce.h>
#include <dlaf/communication/kernels/reduce.h>
#include <dlaf/communication/message.h>
#include <dlaf/communication/rdma.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/sender/traits.h>
#include <dlaf/sender/transform_mpi.h>
#include <dlaf/sender/with_temporary_tile.h>

namespace dlaf::comm {

template <class T, Device D>
[[nodiscard]] dlaf::matrix::ReadWriteTileSender<T, D> scheduleReduceRecvInPlace(
    pika::execution::experimental::unique_any_sender<CommunicatorPipelineExclusiveWrapper> pcomm,
    MPI_Op reduce_op, dlaf::matrix::ReadWriteTileSender<T, D> tile) {
  constexpr Device D_comm = CommunicationDevice_v<D>;

  return internal::scheduleReduceRecvInPlace<D_comm>(std::move(pcomm), reduce_op, std::move(tile));
}

DLAF_EXPAND_ETI_SDCZ_DEVICE(DLAF_SCHEDULE_REDUCE_RECV_IN_PLACE_ETI, );
DLAF_SCHEDULE_REDUCE_RECV_IN_PLACE_ETI(, int, Device::CPU);

template <class T, Device D>
[[nodiscard]] pika::execution::experimental::unique_any_sender<> scheduleReduceSend(
    pika::execution::experimental::unique_any_sender<CommunicatorPipelineExclusiveWrapper> pcomm,
    comm::IndexT_MPI rank_root, MPI_Op reduce_op, dlaf::matrix::ReadOnlyTileSender<T, D> tile) {
  constexpr Device D_comm = CommunicationDevice_v<D>;

  return internal::scheduleReduceSend<D_comm>(std::move(pcomm), rank_root, reduce_op, std::move(tile));
}

DLAF_EXPAND_ETI_SDCZ_DEVICE(DLAF_SCHEDULE_REDUCE_SEND_ETI, );
DLAF_SCHEDULE_REDUCE_SEND_ETI(, int, Device::CPU);
}
