<<<<<<< HEAD
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
namespace broadcast {

/// @brief MPI_Bcast wrapper for sender side accepting a dlaf::comm::Message
template <class T>
void send(Communicator& communicator, Message<T>&& message) {
  MPI_Bcast(const_cast<std::remove_const_t<typename dlaf::comm::Message<T>::element_t>*>(message.data()),
            message.count(), message.mpi_type(), communicator.rank(), communicator);
}

/// @brief MPI_Bcast wrapper for sender side that builds a Message from given trailing arguments
///
/// @param message_args are passed to dlaf::comm::make_message for creating a dlaf::comm::Message
template <class... Ts>
void send(Communicator& communicator, Ts&&... args) {
  sync::broadcast::send(communicator, dlaf::comm::make_message(std::forward<Ts>(args)...));
}

/// @brief MPI_Bcast wrapper for receiver side accepting a dlaf::comm::Message
template <typename T, std::enable_if_t<!std::is_const<T>::value, int> = 0>
void receive_from(int broadcaster_rank, Communicator& communicator, Message<T>&& message) {
  MPI_Bcast(message.data(), message.count(), message.mpi_type(), broadcaster_rank, communicator);
}

/// @brief MPI_Bcast wrapper for receiver side that builds a Message from given trailing arguments
///
/// @param message_args are passed to dlaf::comm::make_message for creating a dlaf::comm::Message
template <class... Ts>
void receive_from(int broadcaster_rank, Communicator& communicator, Ts&&... args) {
  sync::broadcast::receive_from(broadcaster_rank, communicator,
                                dlaf::comm::make_message(std::forward<Ts>(args)...));
}

}
}
}
}
