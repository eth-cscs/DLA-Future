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

#include <dlaf/common/callable_object.h>
#include <dlaf/common/eti.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/communication/kernels/all_reduce.h>
#include <dlaf/communication/kernels/internal/all_reduce.h>
#include <dlaf/communication/message.h>
#include <dlaf/communication/rdma.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/sender/transform_mpi.h>
#include <dlaf/sender/with_temporary_tile.h>

namespace dlaf::comm {

template <class T, Device D_in, Device D_out>
[[nodiscard]] dlaf::matrix::ReadWriteTileSender<T, D_out> scheduleAllReduce(
    pika::execution::experimental::unique_any_sender<CommunicatorPipelineExclusiveWrapper> pcomm,
    MPI_Op reduce_op, dlaf::matrix::ReadOnlyTileSender<T, D_in> tile_in,
    dlaf::matrix::ReadWriteTileSender<T, D_out> tile_out) {
  constexpr Device D_comm_in = CommunicationDevice_v<D_in>;
  constexpr Device D_comm_out = CommunicationDevice_v<D_out>;

  return internal::scheduleAllReduce<D_comm_in, D_comm_out>(std::move(pcomm), reduce_op,
                                                            std::move(tile_in), std::move(tile_out));
}

DLAF_EXPAND_ETI_SDCZ_DEVICE(DLAF_SCHEDULE_ALL_REDUCE_ETI, );
DLAF_SCHEDULE_ALL_REDUCE_ETI(, int, Device::CPU);

template <class T, Device D>
[[nodiscard]] dlaf::matrix::ReadWriteTileSender<T, D> schedule_all_reduce_in_place(
    pika::execution::experimental::unique_any_sender<CommunicatorPipelineExclusiveWrapper> pcomm,
    MPI_Op reduce_op, dlaf::matrix::ReadWriteTileSender<T, D> tile) {
  constexpr Device D_comm = CommunicationDevice_v<D>;

  return internal::schedule_all_reduce_in_place<D_comm>(std::move(pcomm), reduce_op, std::move(tile));
}

DLAF_EXPAND_ETI_SDCZ_DEVICE(DLAF_SCHEDULE_ALL_REDUCE_IN_PLACE_ETI, );
DLAF_SCHEDULE_ALL_REDUCE_IN_PLACE_ETI(, int, Device::CPU);
}
