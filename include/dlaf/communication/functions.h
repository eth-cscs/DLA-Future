//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <functional>

#include "dlaf/communication/communicator.h"
#include "dlaf/communication/mpi_datatypes.h"
#include "dlaf/mpi_header.h"

namespace dlaf {
namespace comm {

/// basic wrapper for MPI_Bcast
template <typename MessageType>
void bcast(int broadcaster_rank, MessageType& message, Communicator communicator) {
  MPI_Bcast(&message, 1, mpi_datatype<MessageType>::type, broadcaster_rank, communicator);
}

/// basic wrapper for MPI_Ibcast
template <typename MessageType>
void async_bcast(int broadcaster_rank, MessageType& message, Communicator communicator,
                 std::function<void()> action_before_retrying) {
  MPI_Request request;
  MPI_Ibcast(&message, 1, mpi_datatype<MessageType>::type, broadcaster_rank, communicator, &request);

  while (true) {
    int test_flag = 1;
    MPI_Test(&request, &test_flag, MPI_STATUS_IGNORE);
    if (test_flag)
      break;
    action_before_retrying();  // has this to be called from both sides of the communication?
  }
}

namespace broadcast {

/// specialized wrapper for MPI_Bcast on sender side
template <typename MessageType>
void send(const MessageType& message, Communicator communicator) {
  bcast(communicator.rank(), const_cast<MessageType&>(message), communicator);
}

/// specialized wrapper for MPI_Bcast on receiver side
template <typename MessageType>
void receive_from(int broadcaster_rank, MessageType& message, Communicator communicator) {
  bcast(broadcaster_rank, message, communicator);
}

}

namespace async_broadcast {

/// specialized wrapper for MPI_Ibcast on sender side
template <typename MessageType>
void send(const MessageType& message, Communicator communicator,
          std::function<void()> action_before_retrying) {
  async_bcast(communicator.rank(), const_cast<MessageType&>(message), communicator,
              action_before_retrying);
}

/// specialized wrapper for MPI_Ibcast on receiver side
template <typename MessageType>
void receive_from(int broadcaster_rank, MessageType& message, Communicator communicator,
                  std::function<void()> action_before_retrying) {
  async_bcast(broadcaster_rank, message, communicator, action_before_retrying);
}

}

}
}
