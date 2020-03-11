//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include "dlaf/common/buffer.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"

namespace dlaf {
namespace comm {
namespace sync {

/// @brief MPI_Send wrapper for sender side accepting a Buffer
///
/// For more information, see the Buffer concept in "dlaf/common/buffer.h"
template <class BufferIn>
void send_to(int receiver_rank, Communicator& communicator, BufferIn&& buffer) {
  int tag = 0;
  auto message = make_message(common::make_buffer(std::forward<BufferIn>(buffer)));
  MPI_Send(message.data(), message.count(), message.mpi_type(), receiver_rank, tag, communicator);
}

/// @brief MPI_Recv wrapper for receiver side accepting a Buffer
///
/// For more information, see the Buffer concept in "dlaf/common/buffer.h"
template <class BufferOut>
void receive_from(int sender_rank, Communicator& communicator, BufferOut&& buffer) {
  int tag = 0;
  auto message = make_message(common::make_buffer(std::forward<BufferOut>(buffer)));
  MPI_Recv(message.data(), message.count(), message.mpi_type(), sender_rank, tag, communicator,
           MPI_STATUS_IGNORE);
}
}
}
}
