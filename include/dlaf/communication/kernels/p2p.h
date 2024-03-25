//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
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

#include <dlaf/common/eti.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/communication/index.h>
#include <dlaf/matrix/tile.h>

namespace dlaf::comm {
template <class T, Device D, class CommSender>
[[nodiscard]] pika::execution::experimental::unique_any_sender<> schedule_send(
    CommSender pcomm, IndexT_MPI dest, IndexT_MPI tag, dlaf::matrix::ReadOnlyTileSender<T, D> tile);

#define DLAF_SCHEDULE_SEND_ETI(kword, Type, Device, CommSender)                    \
  kword template pika::execution::experimental::unique_any_sender<> schedule_send( \
      CommSender pcomm, IndexT_MPI dest, IndexT_MPI tag,                           \
      dlaf::matrix::ReadOnlyTileSender<Type, Device> tile)

// clang-format off
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_SEND_ETI, extern, pika::execution::experimental::unique_any_sender<Communicator>);
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_SEND_ETI, extern, pika::execution::experimental::any_sender<Communicator>);
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_SEND_ETI, extern, CommunicatorPipelineSharedSender);
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_SEND_ETI, extern, CommunicatorPipelineExclusiveSender);
// clang-format on

template <class T, Device D, class CommSender>
[[nodiscard]] dlaf::matrix::ReadWriteTileSender<T, D> schedule_recv(
    CommSender pcomm, IndexT_MPI source, IndexT_MPI tag, dlaf::matrix::ReadWriteTileSender<T, D> tile);

#define DLAF_SCHEDULE_RECV_ETI(kword, Type, Device, CommSender)                 \
  kword template dlaf::matrix::ReadWriteTileSender<Type, Device> schedule_recv( \
      CommSender pcomm, IndexT_MPI source, IndexT_MPI tag,                      \
      dlaf::matrix::ReadWriteTileSender<Type, Device> tile)

// clang-format off
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_RECV_ETI, extern, pika::execution::experimental::unique_any_sender<Communicator>);
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_RECV_ETI, extern, pika::execution::experimental::any_sender<Communicator>);
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_RECV_ETI, extern, CommunicatorPipelineSharedSender);
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_RECV_ETI, extern, CommunicatorPipelineExclusiveSender);
// clang-format on
}
