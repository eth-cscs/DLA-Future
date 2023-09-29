//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <complex>

#include <mpi.h>

#include <pika/execution.hpp>

#include <dlaf/common/data.h>
#include <dlaf/common/eti.h>
#include <dlaf/common/pipeline.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/message.h>
#include <dlaf/matrix/tile.h>

namespace dlaf::comm {
/// Schedule an in-place reduction receive.
///
/// The returned sender signals completion when the receive is done. The input
/// sender tile must be writable so that the received data can be written to it.
/// The input tile is sent by the returned sender.
template <class T, Device D>
[[nodiscard]] dlaf::matrix::ReadWriteTileSender<T, D> scheduleReduceRecvInPlace(
    pika::execution::experimental::unique_any_sender<common::Pipeline<Communicator>::Wrapper> pcomm,
    MPI_Op reduce_op, dlaf::matrix::ReadWriteTileSender<T, D> tile);

#define DLAF_SCHEDULE_REDUCE_RECV_IN_PLACE_ETI(kword, Type, Device)                                    \
  kword template dlaf::matrix::ReadWriteTileSender<Type, Device> scheduleReduceRecvInPlace(            \
      pika::execution::experimental::unique_any_sender<common::Pipeline<Communicator>::Wrapper> pcomm, \
      MPI_Op reduce_op, dlaf::matrix::ReadWriteTileSender<Type, Device> tile)

DLAF_EXPAND_ETI_SDCZ_DEVICE(DLAF_SCHEDULE_REDUCE_RECV_IN_PLACE_ETI, extern);
DLAF_SCHEDULE_REDUCE_RECV_IN_PLACE_ETI(extern, int, Device::CPU);

/// Schedule a reduction send.
///
/// The returned sender signals completion when the send is done. If the input
/// tile is movable it will be sent by the returned sender. Otherwise a void
/// sender is returned.
template <class T, Device D>
[[nodiscard]] pika::execution::experimental::unique_any_sender<> scheduleReduceSend(
    pika::execution::experimental::unique_any_sender<common::Pipeline<Communicator>::Wrapper> pcomm,
    comm::IndexT_MPI rank_root, MPI_Op reduce_op, dlaf::matrix::ReadOnlyTileSender<T, D> tile);

#define DLAF_SCHEDULE_REDUCE_SEND_ETI(kword, Type, Device)                                             \
  kword template pika::execution::experimental::unique_any_sender<> scheduleReduceSend(                \
      pika::execution::experimental::unique_any_sender<common::Pipeline<Communicator>::Wrapper> pcomm, \
      comm::IndexT_MPI rank_root, MPI_Op reduce_op,                                                    \
      dlaf::matrix::ReadOnlyTileSender<Type, Device> tile)

DLAF_EXPAND_ETI_SDCZ_DEVICE(DLAF_SCHEDULE_REDUCE_SEND_ETI, extern);
DLAF_SCHEDULE_REDUCE_SEND_ETI(extern, int, Device::CPU);
}
