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

#include "dlaf/common/callable_object.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/traits.h"
#include "dlaf/sender/transform_mpi.h"
#include "dlaf/sender/when_all_lift.h"

namespace dlaf::comm {

namespace internal {

// Non-blocking point to point send
template <class T, Device D>
void send(const Communicator& comm, const matrix::Tile<const T, D>& tile, IndexT_MPI dest,
          IndexT_MPI tag, MPI_Request* req) {
#if !defined(DLAF_WITH_CUDA_RDMA)
  static_assert(D == Device::CPU, "DLAF_WITH_CUDA_RDMA=off, MPI accepts just CPU memory.");
#endif

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(
      MPI_Isend(const_cast<T*>(msg.data()), msg.count(), msg.mpi_type(), dest, tag, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(send);

// Non-blocking point to point receive
template <class T, Device D>
auto recv(const Communicator& comm, const matrix::Tile<T, D>& tile, IndexT_MPI source, IndexT_MPI tag,
          MPI_Request* req) {
#if !defined(DLAF_WITH_CUDA_RDMA)
  static_assert(D == Device::CPU, "DLAF_WITH_CUDA_RDMA=off, MPI accepts just CPU memory.");
#endif

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(MPI_Irecv(msg.data(), msg.count(), msg.mpi_type(), source, tag, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(recv);
}

template <class CommSender, class Sender>
void scheduleSend(IndexT_MPI dest, CommSender&& pcomm, IndexT_MPI tag, Sender&& tile) {
  using dlaf::internal::whenAllLift;
  using pika::execution::experimental::start_detached;

  start_detached(whenAllLift(std::forward<CommSender>(pcomm), std::forward<Sender>(tile), dest, tag) |
                 internal::transformMPI(internal::send_o));
}

template <class CommSender, class Sender>
auto scheduleRecv(IndexT_MPI source, CommSender&& pcomm, IndexT_MPI tag, Sender&& tile) {
  using dlaf::internal::whenAllLift;
  using pika::execution::experimental::start_detached;

  start_detached(whenAllLift(std::forward<CommSender>(pcomm), std::forward<Sender>(tile), source, tag) |
                 internal::transformMPI(internal::recv_o));
}
}
