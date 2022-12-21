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

#include <complex>

#include <mpi.h>
#include <pika/execution.hpp>

#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/matrix/tile.h"

namespace dlaf::comm {
template <class T, Device D, class Comm>
[[nodiscard]] pika::execution::experimental::unique_any_sender<> scheduleSend(
    pika::execution::experimental::unique_any_sender<Comm> pcomm, IndexT_MPI dest, IndexT_MPI tag,
    pika::execution::experimental::unique_any_sender<pika::shared_future<matrix::Tile<const T, D>>> tile);

#define DLAF_SCHEDULE_SEND_SFTILE_ETI(kword, Type, Device, Comm)                              \
  kword template pika::execution::experimental::unique_any_sender<>                           \
  scheduleSend(pika::execution::experimental::unique_any_sender<Comm> pcomm, IndexT_MPI dest, \
               IndexT_MPI tag,                                                                \
               pika::execution::experimental::unique_any_sender<                              \
                   pika::shared_future<matrix::Tile<const Type, Device>>>                     \
                   tile)

DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, float, Device::CPU, Communicator);
DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, double, Device::CPU, Communicator);
DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, std::complex<float>, Device::CPU, Communicator);
DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, std::complex<double>, Device::CPU, Communicator);

DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, float, Device::CPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, double, Device::CPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, std::complex<float>, Device::CPU,
                              common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, std::complex<double>, Device::CPU,
                              common::PromiseGuard<Communicator>);

#ifdef DLAF_WITH_GPU
DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, float, Device::GPU, Communicator);
DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, double, Device::GPU, Communicator);
DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, std::complex<float>, Device::GPU, Communicator);
DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, std::complex<double>, Device::GPU, Communicator);

DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, float, Device::GPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, double, Device::GPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, std::complex<float>, Device::GPU,
                              common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, std::complex<double>, Device::GPU,
                              common::PromiseGuard<Communicator>);
#endif

template <class T, Device D, class Comm>
[[nodiscard]] pika::execution::experimental::unique_any_sender<> scheduleSend(
    pika::execution::experimental::unique_any_sender<Comm> pcomm, IndexT_MPI dest, IndexT_MPI tag,
    pika::execution::experimental::unique_any_sender<matrix::tile_async_ro_mutex_wrapper_type<T, D>>
        tile);

#define DLAF_SCHEDULE_SEND_WRAPPER_ETI(kword, Type, Device, Comm)                             \
  kword template pika::execution::experimental::unique_any_sender<>                           \
  scheduleSend(pika::execution::experimental::unique_any_sender<Comm> pcomm, IndexT_MPI dest, \
               IndexT_MPI tag,                                                                \
               pika::execution::experimental::unique_any_sender<                              \
                   matrix::tile_async_ro_mutex_wrapper_type<Type, Device>>                    \
                   tile)

DLAF_SCHEDULE_SEND_WRAPPER_ETI(extern, float, Device::CPU, Communicator);
DLAF_SCHEDULE_SEND_WRAPPER_ETI(extern, double, Device::CPU, Communicator);
DLAF_SCHEDULE_SEND_WRAPPER_ETI(extern, std::complex<float>, Device::CPU, Communicator);
DLAF_SCHEDULE_SEND_WRAPPER_ETI(extern, std::complex<double>, Device::CPU, Communicator);

DLAF_SCHEDULE_SEND_WRAPPER_ETI(extern, float, Device::CPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_WRAPPER_ETI(extern, double, Device::CPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_WRAPPER_ETI(extern, std::complex<float>, Device::CPU,
                               common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_WRAPPER_ETI(extern, std::complex<double>, Device::CPU,
                               common::PromiseGuard<Communicator>);

#ifdef DLAF_WITH_GPU
DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, float, Device::GPU, Communicator);
DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, double, Device::GPU, Communicator);
DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, std::complex<float>, Device::GPU, Communicator);
DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, std::complex<double>, Device::GPU, Communicator);

DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, float, Device::GPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, double, Device::GPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, std::complex<float>, Device::GPU,
                              common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_SEND_SFTILE_ETI(extern, std::complex<double>, Device::GPU,
                              common::PromiseGuard<Communicator>);
#endif

template <class T, Device D, class Comm>
[[nodiscard]] pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> scheduleRecv(
    pika::execution::experimental::unique_any_sender<Comm> pcomm, IndexT_MPI source, IndexT_MPI tag,
    pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> tile);

#define DLAF_SCHEDULE_RECV_ETI(kword, Type, Device, Comm)                                       \
  kword template pika::execution::experimental::unique_any_sender<matrix::Tile<Type, Device>>   \
  scheduleRecv(pika::execution::experimental::unique_any_sender<Comm> pcomm, IndexT_MPI source, \
               IndexT_MPI tag,                                                                  \
               pika::execution::experimental::unique_any_sender<matrix::Tile<Type, Device>> tile)

DLAF_SCHEDULE_RECV_ETI(extern, float, Device::CPU, Communicator);
DLAF_SCHEDULE_RECV_ETI(extern, double, Device::CPU, Communicator);
DLAF_SCHEDULE_RECV_ETI(extern, std::complex<float>, Device::CPU, Communicator);
DLAF_SCHEDULE_RECV_ETI(extern, std::complex<double>, Device::CPU, Communicator);

DLAF_SCHEDULE_RECV_ETI(extern, float, Device::CPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_RECV_ETI(extern, double, Device::CPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_RECV_ETI(extern, std::complex<float>, Device::CPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_RECV_ETI(extern, std::complex<double>, Device::CPU, common::PromiseGuard<Communicator>);

#ifdef DLAF_WITH_GPU
DLAF_SCHEDULE_RECV_ETI(extern, float, Device::GPU, Communicator);
DLAF_SCHEDULE_RECV_ETI(extern, double, Device::GPU, Communicator);
DLAF_SCHEDULE_RECV_ETI(extern, std::complex<float>, Device::GPU, Communicator);
DLAF_SCHEDULE_RECV_ETI(extern, std::complex<double>, Device::GPU, Communicator);

DLAF_SCHEDULE_RECV_ETI(extern, float, Device::GPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_RECV_ETI(extern, double, Device::GPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_RECV_ETI(extern, std::complex<float>, Device::GPU, common::PromiseGuard<Communicator>);
DLAF_SCHEDULE_RECV_ETI(extern, std::complex<double>, Device::GPU, common::PromiseGuard<Communicator>);
#endif
}
