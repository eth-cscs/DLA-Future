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

#include <dlaf/common/pipeline.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/matrix/tile.h>

namespace dlaf::comm {
template <class T, Device D, class CommSender>
[[nodiscard]] pika::execution::experimental::unique_any_sender<> scheduleSend(
    CommSender pcomm, IndexT_MPI dest, IndexT_MPI tag, dlaf::matrix::ReadOnlyTileSender<T, D> tile);

#define DLAF_SCHEDULE_SEND_ETI(kword, Type, Device, CommSender)                   \
  kword template pika::execution::experimental::unique_any_sender<> scheduleSend( \
      CommSender pcomm, IndexT_MPI dest, IndexT_MPI tag,                          \
      dlaf::matrix::ReadOnlyTileSender<Type, Device> tile)

// clang-format off
DLAF_SCHEDULE_SEND_ETI(extern, float, Device::CPU, pika::execution::experimental::unique_any_sender<Communicator>);
DLAF_SCHEDULE_SEND_ETI(extern, double, Device::CPU, pika::execution::experimental::unique_any_sender<Communicator>);
DLAF_SCHEDULE_SEND_ETI(extern, std::complex<float>, Device::CPU, pika::execution::experimental::unique_any_sender<Communicator>);
DLAF_SCHEDULE_SEND_ETI(extern, std::complex<double>, Device::CPU, pika::execution::experimental::unique_any_sender<Communicator>);

DLAF_SCHEDULE_SEND_ETI(extern, float, Device::CPU, common::Pipeline<Communicator>::ReadOnlySender);
DLAF_SCHEDULE_SEND_ETI(extern, double, Device::CPU, common::Pipeline<Communicator>::ReadOnlySender);
DLAF_SCHEDULE_SEND_ETI(extern, std::complex<float>, Device::CPU, common::Pipeline<Communicator>::ReadOnlySender);
DLAF_SCHEDULE_SEND_ETI(extern, std::complex<double>, Device::CPU, common::Pipeline<Communicator>::ReadOnlySender);

DLAF_SCHEDULE_SEND_ETI(extern, float, Device::CPU, common::Pipeline<Communicator>::ReadWriteSender);
DLAF_SCHEDULE_SEND_ETI(extern, double, Device::CPU, common::Pipeline<Communicator>::ReadWriteSender);
DLAF_SCHEDULE_SEND_ETI(extern, std::complex<float>, Device::CPU, common::Pipeline<Communicator>::ReadWriteSender);
DLAF_SCHEDULE_SEND_ETI(extern, std::complex<double>, Device::CPU, common::Pipeline<Communicator>::ReadWriteSender);

#ifdef DLAF_WITH_GPU
DLAF_SCHEDULE_SEND_ETI(extern, float, Device::GPU, pika::execution::experimental::unique_any_sender<Communicator>);
DLAF_SCHEDULE_SEND_ETI(extern, double, Device::GPU, pika::execution::experimental::unique_any_sender<Communicator>);
DLAF_SCHEDULE_SEND_ETI(extern, std::complex<float>, Device::GPU, pika::execution::experimental::unique_any_sender<Communicator>);
DLAF_SCHEDULE_SEND_ETI(extern, std::complex<double>, Device::GPU, pika::execution::experimental::unique_any_sender<Communicator>);

DLAF_SCHEDULE_SEND_ETI(extern, float, Device::GPU, common::Pipeline<Communicator>::ReadOnlySender);
DLAF_SCHEDULE_SEND_ETI(extern, double, Device::GPU, common::Pipeline<Communicator>::ReadOnlySender);
DLAF_SCHEDULE_SEND_ETI(extern, std::complex<float>, Device::GPU, common::Pipeline<Communicator>::ReadOnlySender);
DLAF_SCHEDULE_SEND_ETI(extern, std::complex<double>, Device::GPU, common::Pipeline<Communicator>::ReadOnlySender);

DLAF_SCHEDULE_SEND_ETI(extern, float, Device::GPU, common::Pipeline<Communicator>::ReadWriteSender);
DLAF_SCHEDULE_SEND_ETI(extern, double, Device::GPU, common::Pipeline<Communicator>::ReadWriteSender);
DLAF_SCHEDULE_SEND_ETI(extern, std::complex<float>, Device::GPU, common::Pipeline<Communicator>::ReadWriteSender);
DLAF_SCHEDULE_SEND_ETI(extern, std::complex<double>, Device::GPU, common::Pipeline<Communicator>::ReadWriteSender);
#endif
// clang-format on

template <class T, Device D, class CommSender>
[[nodiscard]] dlaf::matrix::ReadWriteTileSender<T, D> scheduleRecv(
    CommSender pcomm, IndexT_MPI source, IndexT_MPI tag, dlaf::matrix::ReadWriteTileSender<T, D> tile);

#define DLAF_SCHEDULE_RECV_ETI(kword, Type, Device, CommSender)                \
  kword template dlaf::matrix::ReadWriteTileSender<Type, Device> scheduleRecv( \
      CommSender pcomm, IndexT_MPI source, IndexT_MPI tag,                     \
      dlaf::matrix::ReadWriteTileSender<Type, Device> tile)

// clang-format off
DLAF_SCHEDULE_RECV_ETI(extern, float, Device::CPU, pika::execution::experimental::unique_any_sender<Communicator>);
DLAF_SCHEDULE_RECV_ETI(extern, double, Device::CPU, pika::execution::experimental::unique_any_sender<Communicator>);
DLAF_SCHEDULE_RECV_ETI(extern, std::complex<float>, Device::CPU, pika::execution::experimental::unique_any_sender<Communicator>);
DLAF_SCHEDULE_RECV_ETI(extern, std::complex<double>, Device::CPU, pika::execution::experimental::unique_any_sender<Communicator>);

DLAF_SCHEDULE_RECV_ETI(extern, float, Device::CPU, common::Pipeline<Communicator>::ReadOnlySender);
DLAF_SCHEDULE_RECV_ETI(extern, double, Device::CPU, common::Pipeline<Communicator>::ReadOnlySender);
DLAF_SCHEDULE_RECV_ETI(extern, std::complex<float>, Device::CPU, common::Pipeline<Communicator>::ReadOnlySender);
DLAF_SCHEDULE_RECV_ETI(extern, std::complex<double>, Device::CPU, common::Pipeline<Communicator>::ReadOnlySender);

DLAF_SCHEDULE_RECV_ETI(extern, float, Device::CPU, common::Pipeline<Communicator>::ReadWriteSender);
DLAF_SCHEDULE_RECV_ETI(extern, double, Device::CPU, common::Pipeline<Communicator>::ReadWriteSender);
DLAF_SCHEDULE_RECV_ETI(extern, std::complex<float>, Device::CPU, common::Pipeline<Communicator>::ReadWriteSender);
DLAF_SCHEDULE_RECV_ETI(extern, std::complex<double>, Device::CPU, common::Pipeline<Communicator>::ReadWriteSender);

#ifdef DLAF_WITH_GPU
DLAF_SCHEDULE_RECV_ETI(extern, float, Device::GPU, pika::execution::experimental::unique_any_sender<Communicator>);
DLAF_SCHEDULE_RECV_ETI(extern, double, Device::GPU, pika::execution::experimental::unique_any_sender<Communicator>);
DLAF_SCHEDULE_RECV_ETI(extern, std::complex<float>, Device::GPU, pika::execution::experimental::unique_any_sender<Communicator>);
DLAF_SCHEDULE_RECV_ETI(extern, std::complex<double>, Device::GPU, pika::execution::experimental::unique_any_sender<Communicator>);

DLAF_SCHEDULE_RECV_ETI(extern, float, Device::GPU, common::Pipeline<Communicator>::ReadOnlySender);
DLAF_SCHEDULE_RECV_ETI(extern, double, Device::GPU, common::Pipeline<Communicator>::ReadOnlySender);
DLAF_SCHEDULE_RECV_ETI(extern, std::complex<float>, Device::GPU, common::Pipeline<Communicator>::ReadOnlySender);
DLAF_SCHEDULE_RECV_ETI(extern, std::complex<double>, Device::GPU, common::Pipeline<Communicator>::ReadOnlySender);

DLAF_SCHEDULE_RECV_ETI(extern, float, Device::GPU, common::Pipeline<Communicator>::ReadWriteSender);
DLAF_SCHEDULE_RECV_ETI(extern, double, Device::GPU, common::Pipeline<Communicator>::ReadWriteSender);
DLAF_SCHEDULE_RECV_ETI(extern, std::complex<float>, Device::GPU, common::Pipeline<Communicator>::ReadWriteSender);
DLAF_SCHEDULE_RECV_ETI(extern, std::complex<double>, Device::GPU, common::Pipeline<Communicator>::ReadWriteSender);
#endif
// clang-format on
}
