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

#include "dlaf/common/data.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"
#include "dlaf/matrix/tile.h"

namespace dlaf::comm {
/// Schedule an in-place reduction receive.
///
/// The returned sender signals completion when the receive is done. The input
/// sender tile must be writable so that the received data can be written to it.
/// The input tile is sent by the returned sender.
template <class T, Device D>
[[nodiscard]] pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> scheduleReduceRecvInPlace(
    pika::execution::experimental::unique_any_sender<dlaf::common::PromiseGuard<Communicator>> pcomm,
    MPI_Op reduce_op, pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> tile);

#define DLAF_SCHEDULE_REDUCE_RECV_IN_PLACE_ETI(kword, Type, Device)                                      \
  kword template pika::execution::experimental::unique_any_sender<matrix::Tile<Type, Device>>            \
  scheduleReduceRecvInPlace(pika::execution::experimental::unique_any_sender<                            \
                                dlaf::common::PromiseGuard<Communicator>>                                \
                                pcomm,                                                                   \
                            MPI_Op reduce_op,                                                            \
                            pika::execution::experimental::unique_any_sender<matrix::Tile<Type, Device>> \
                                tile)

DLAF_SCHEDULE_REDUCE_RECV_IN_PLACE_ETI(extern, int, Device::CPU);
DLAF_SCHEDULE_REDUCE_RECV_IN_PLACE_ETI(extern, float, Device::CPU);
DLAF_SCHEDULE_REDUCE_RECV_IN_PLACE_ETI(extern, double, Device::CPU);
DLAF_SCHEDULE_REDUCE_RECV_IN_PLACE_ETI(extern, std::complex<float>, Device::CPU);
DLAF_SCHEDULE_REDUCE_RECV_IN_PLACE_ETI(extern, std::complex<double>, Device::CPU);

#ifdef DLAF_WITH_GPU
DLAF_SCHEDULE_REDUCE_RECV_IN_PLACE_ETI(extern, float, Device::GPU);
DLAF_SCHEDULE_REDUCE_RECV_IN_PLACE_ETI(extern, double, Device::GPU);
DLAF_SCHEDULE_REDUCE_RECV_IN_PLACE_ETI(extern, std::complex<float>, Device::GPU);
DLAF_SCHEDULE_REDUCE_RECV_IN_PLACE_ETI(extern, std::complex<double>, Device::GPU);
#endif

/// Schedule a reduction send.
///
/// The returned sender signals completion when the send is done. If the input
/// tile is movable it will be sent by the returned sender. Otherwise a void
/// sender is returned.
template <class T, Device D>
[[nodiscard]] pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> scheduleReduceSend(
    pika::execution::experimental::unique_any_sender<dlaf::common::PromiseGuard<Communicator>> pcomm,
    comm::IndexT_MPI rank_root, MPI_Op reduce_op,
    pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> tile);

#define DLAF_SCHEDULE_REDUCE_SEND_ETI(kword, Type, Device)                                    \
  kword template pika::execution::experimental::unique_any_sender<matrix::Tile<Type, Device>> \
  scheduleReduceSend(pika::execution::experimental::unique_any_sender<                        \
                         dlaf::common::PromiseGuard<Communicator>>                            \
                         pcomm,                                                               \
                     comm::IndexT_MPI rank_root, MPI_Op reduce_op,                            \
                     pika::execution::experimental::unique_any_sender<matrix::Tile<Type, Device>> tile)

DLAF_SCHEDULE_REDUCE_SEND_ETI(extern, float, Device::CPU);
DLAF_SCHEDULE_REDUCE_SEND_ETI(extern, double, Device::CPU);
DLAF_SCHEDULE_REDUCE_SEND_ETI(extern, std::complex<float>, Device::CPU);
DLAF_SCHEDULE_REDUCE_SEND_ETI(extern, std::complex<double>, Device::CPU);

#ifdef DLAF_WITH_GPU
DLAF_SCHEDULE_REDUCE_SEND_ETI(extern, float, Device::GPU);
DLAF_SCHEDULE_REDUCE_SEND_ETI(extern, double, Device::GPU);
DLAF_SCHEDULE_REDUCE_SEND_ETI(extern, std::complex<float>, Device::GPU);
DLAF_SCHEDULE_REDUCE_SEND_ETI(extern, std::complex<double>, Device::GPU);
#endif

/// \overload scheduleReduceSend
template <class T, Device D>
[[nodiscard]] pika::execution::experimental::unique_any_sender<> scheduleReduceSend(
    pika::execution::experimental::unique_any_sender<dlaf::common::PromiseGuard<Communicator>> pcomm,
    comm::IndexT_MPI rank_root, MPI_Op reduce_op,
    pika::execution::experimental::unique_any_sender<pika::shared_future<matrix::Tile<const T, D>>> tile);

#define DLAF_SCHEDULE_REDUCE_SEND_SFTILE_ETI(kword, Type, Device)               \
  kword template pika::execution::experimental::unique_any_sender<>             \
  scheduleReduceSend(pika::execution::experimental::unique_any_sender<          \
                         dlaf::common::PromiseGuard<Communicator>>              \
                         pcomm,                                                 \
                     comm::IndexT_MPI rank_root, MPI_Op reduce_op,              \
                     pika::execution::experimental::unique_any_sender<          \
                         pika::shared_future<matrix::Tile<const Type, Device>>> \
                         tile)

DLAF_SCHEDULE_REDUCE_SEND_SFTILE_ETI(extern, int, Device::CPU);
DLAF_SCHEDULE_REDUCE_SEND_SFTILE_ETI(extern, float, Device::CPU);
DLAF_SCHEDULE_REDUCE_SEND_SFTILE_ETI(extern, double, Device::CPU);
DLAF_SCHEDULE_REDUCE_SEND_SFTILE_ETI(extern, std::complex<float>, Device::CPU);
DLAF_SCHEDULE_REDUCE_SEND_SFTILE_ETI(extern, std::complex<double>, Device::CPU);

#ifdef DLAF_WITH_GPU
DLAF_SCHEDULE_REDUCE_SEND_SFTILE_ETI(extern, float, Device::GPU);
DLAF_SCHEDULE_REDUCE_SEND_SFTILE_ETI(extern, double, Device::GPU);
DLAF_SCHEDULE_REDUCE_SEND_SFTILE_ETI(extern, std::complex<float>, Device::GPU);
DLAF_SCHEDULE_REDUCE_SEND_SFTILE_ETI(extern, std::complex<double>, Device::GPU);
#endif

/// \overload scheduleReduceSend
template <class T, Device D>
[[nodiscard]] pika::execution::experimental::unique_any_sender<> scheduleReduceSend(
    pika::execution::experimental::unique_any_sender<dlaf::common::PromiseGuard<Communicator>> pcomm,
    comm::IndexT_MPI rank_root, MPI_Op reduce_op,
    pika::execution::experimental::unique_any_sender<matrix::tile_async_ro_mutex_wrapper_type<T, D>>
        tile);

#define DLAF_SCHEDULE_REDUCE_SEND_WRAPPER_ETI(kword, Type, Device)               \
  kword template pika::execution::experimental::unique_any_sender<>              \
  scheduleReduceSend(pika::execution::experimental::unique_any_sender<           \
                         dlaf::common::PromiseGuard<Communicator>>               \
                         pcomm,                                                  \
                     comm::IndexT_MPI rank_root, MPI_Op reduce_op,               \
                     pika::execution::experimental::unique_any_sender<           \
                         matrix::tile_async_ro_mutex_wrapper_type<Type, Device>> \
                         tile)

DLAF_SCHEDULE_REDUCE_SEND_WRAPPER_ETI(extern, int, Device::CPU);
DLAF_SCHEDULE_REDUCE_SEND_WRAPPER_ETI(extern, float, Device::CPU);
DLAF_SCHEDULE_REDUCE_SEND_WRAPPER_ETI(extern, double, Device::CPU);
DLAF_SCHEDULE_REDUCE_SEND_WRAPPER_ETI(extern, std::complex<float>, Device::CPU);
DLAF_SCHEDULE_REDUCE_SEND_WRAPPER_ETI(extern, std::complex<double>, Device::CPU);

#ifdef DLAF_WITH_GPU
DLAF_SCHEDULE_REDUCE_SEND_WRAPPER_ETI(extern, float, Device::GPU);
DLAF_SCHEDULE_REDUCE_SEND_WRAPPER_ETI(extern, double, Device::GPU);
DLAF_SCHEDULE_REDUCE_SEND_WRAPPER_ETI(extern, std::complex<float>, Device::GPU);
DLAF_SCHEDULE_REDUCE_SEND_WRAPPER_ETI(extern, std::complex<double>, Device::GPU);
#endif
}
