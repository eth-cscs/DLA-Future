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
/// Schedule a broadcast send.
///
/// The returned sender signals completion when the send is done. If the input
/// tile is movable it will be sent by the returned sender. Otherwise a void
/// sender is returned.
template <class T, Device D>
[[nodiscard]] pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> scheduleSendBcast(
    pika::execution::experimental::unique_any_sender<dlaf::common::PromiseGuard<Communicator>> pcomm,
    pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> tile);

#define DLAF_SCHEDULE_SEND_BCAST_ETI(kword, Type, Device)                                     \
  kword template pika::execution::experimental::unique_any_sender<matrix::Tile<Type, Device>> \
  scheduleSendBcast(pika::execution::experimental::unique_any_sender<                         \
                        dlaf::common::PromiseGuard<Communicator>>                             \
                        pcomm,                                                                \
                    pika::execution::experimental::unique_any_sender<matrix::Tile<Type, Device>> tile)

DLAF_SCHEDULE_SEND_BCAST_ETI(extern, float, Device::CPU);
DLAF_SCHEDULE_SEND_BCAST_ETI(extern, double, Device::CPU);
DLAF_SCHEDULE_SEND_BCAST_ETI(extern, std::complex<float>, Device::CPU);
DLAF_SCHEDULE_SEND_BCAST_ETI(extern, std::complex<double>, Device::CPU);

template <class T, Device D>
[[nodiscard]] pika::execution::experimental::unique_any_sender<> scheduleSendBcast(
    pika::execution::experimental::unique_any_sender<dlaf::common::PromiseGuard<Communicator>> pcomm,
    pika::execution::experimental::unique_any_sender<pika::shared_future<matrix::Tile<const T, D>>> tile);

#define DLAF_SCHEDULE_SEND_BCAST_SFTILE_ETI(kword, Type, Device)               \
  kword template pika::execution::experimental::unique_any_sender<>            \
  scheduleSendBcast(pika::execution::experimental::unique_any_sender<          \
                        dlaf::common::PromiseGuard<Communicator>>              \
                        pcomm,                                                 \
                    pika::execution::experimental::unique_any_sender<          \
                        pika::shared_future<matrix::Tile<const Type, Device>>> \
                        tile)

DLAF_SCHEDULE_SEND_BCAST_SFTILE_ETI(extern, float, Device::CPU);
DLAF_SCHEDULE_SEND_BCAST_SFTILE_ETI(extern, double, Device::CPU);
DLAF_SCHEDULE_SEND_BCAST_SFTILE_ETI(extern, std::complex<float>, Device::CPU);
DLAF_SCHEDULE_SEND_BCAST_SFTILE_ETI(extern, std::complex<double>, Device::CPU);

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
