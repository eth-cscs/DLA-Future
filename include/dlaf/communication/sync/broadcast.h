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

#include "dlaf/common/buffer.h"

namespace dlaf {
namespace comm {
namespace sync {
namespace broadcast {

/// @brief MPI_Bcast wrapper for sender side accepting a Buffer
///
/// For more information, see the Buffer concept in "dlaf/common/buffer.h"
template <class BufferIn>
void send(Communicator& communicator, BufferIn&& message_to_send) {
  auto buffer = common::make_buffer(message_to_send);
  using DataT = std::remove_const_t<typename common::buffer_traits<decltype(buffer)>::element_t>;

  auto message = comm::make_message(std::move(buffer));
  MPI_Bcast(const_cast<DataT*>(message.data()), message.count(), message.mpi_type(), communicator.rank(),
            communicator);
}

/// @brief MPI_Bcast wrapper for receiver side accepting a dlaf::comm::Message
///
/// For more information, see the Buffer concept in "dlaf/common/buffer.h"
template <class BufferOut>
void receive_from(int broadcaster_rank, Communicator& communicator, BufferOut&& buffer) {
  auto message = comm::make_message(common::make_buffer(std::forward<BufferOut>(buffer)));
  MPI_Bcast(message.data(), message.count(), message.mpi_type(), broadcaster_rank, communicator);
}
}
}
}
}
