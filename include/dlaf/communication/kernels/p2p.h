//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
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

namespace dlaf {
namespace comm {

namespace internal {

// Non-blocking point to point send
template <class T, Device D>
void send(const matrix::Tile<const T, D>& tile, IndexT_MPI receiver, IndexT_MPI tag,
          common::PromiseGuard<Communicator> pcomm, MPI_Request* req) {
  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(MPI_Isend(const_cast<T*>(msg.data()), msg.count(), msg.mpi_type(), receiver, tag,
                                 pcomm.ref(), req));
}

DLAF_MAKE_CALLABLE_OBJECT(send);

// Non-blocking point to point receive
template <class T, Device D>
auto recv(const matrix::Tile<T, D>& tile, IndexT_MPI sender, IndexT_MPI tag,
          common::PromiseGuard<Communicator> pcomm, MPI_Request* req) {
  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(
      MPI_Irecv(msg.data(), msg.count(), msg.mpi_type(), sender, tag, pcomm.ref(), req));
}

DLAF_MAKE_CALLABLE_OBJECT(recv);
}

template <class CommSender, class Sender>
void scheduleSend(IndexT_MPI receiver, CommSender&& pcomm, IndexT_MPI tag, Sender tile) {
  using dlaf::internal::whenAllLift;
  using pika::execution::experimental::start_detached;

  whenAllLift(std::move(tile), receiver, tag, std::forward<CommSender>(pcomm)) |
      internal::transformMPI(internal::send_o) | pika::execution::experimental::start_detached();
}

template <class CommSender, class Sender>
auto scheduleRecv(IndexT_MPI sender, CommSender&& pcomm, IndexT_MPI tag, Sender tile) {
  using dlaf::internal::whenAllLift;
  using pika::execution::experimental::start_detached;

  whenAllLift(std::move(tile), sender, tag, std::forward<CommSender>(pcomm)) |
      internal::transformMPI(internal::recv_o) | pika::execution::experimental::start_detached();
}
}
}
