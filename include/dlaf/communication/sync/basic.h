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

#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"

namespace dlaf {
namespace comm {
namespace sync {

/// @brief MPI_Send wrapper accepting a dlaf::comm::Message
template <class T>
void send_to(int receiver_rank, Communicator& communicator, Message<T>&& message) {
  int tag = 0;
  MPI_Send(message.data(), message.count(), message.mpi_type(), receiver_rank, tag, communicator);
}

/// @brief MPI_Send wrapper that builds a Message from given trailing arguments
///
/// @param message_args are passed to dlaf::comm::make_message for creating a dlaf::comm::Message
template <class... Ts>
void send_to(int receiver_rank, Communicator& communicator, Ts&&... message_args) {
  sync::send_to(receiver_rank, communicator,
                dlaf::comm::make_message(std::forward<Ts>(message_args)...));
}

/// @brief MPI_Recv wrapper accepting a dlaf::comm::Message
template <class T>
void receive_from(int sender_rank, Communicator& communicator, Message<T>&& message) {
  int tag = 0;
  MPI_Recv(message.data(), message.count(), message.mpi_type(), sender_rank, tag, communicator,
           MPI_STATUS_IGNORE);
}

/// @brief MPI_Recv wrapper that builds a Message from given trailing arguments
///
/// @param message_args are passed to dlaf::comm::make_message for creating a dlaf::comm::Message
template <class... Ts>
void receive_from(int sender_rank, Communicator& communicator, Ts&&... message_args) {
  sync::receive_from(sender_rank, communicator,
                     dlaf::comm::make_message(std::forward<Ts>(message_args)...));
}

}
}
}
