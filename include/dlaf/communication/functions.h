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
#include "dlaf/communication/message.h"

namespace dlaf {
namespace comm {

namespace internal {
template <typename T>
void async_bcast(int broadcaster_rank, T* ptr, std::size_t size, MPI_Datatype mpi_type,
                 Communicator communicator, std::function<void()> action_before_retrying) {
  MPI_Request request;
  MPI_Ibcast(const_cast<std::remove_const_t<T>*>(ptr), size, mpi_type, broadcaster_rank, communicator,
             &request);

  while (true) {
    int test_flag = 1;
    MPI_Test(&request, &test_flag, MPI_STATUS_IGNORE);
    if (test_flag)
      break;
    action_before_retrying();  // has this to be called from both sides of the communication?
  }
}
}

namespace broadcast {

/// specialized wrapper for MPI_Bcast on sender side
template <class T>
void send(message<T>&& message, Communicator& communicator) {
  MPI_Bcast(const_cast<std::remove_const_t<typename dlaf::comm::message<T>::T>*>(message.ptr()),
            message.count(), message.mpi_type(), communicator.rank(), communicator);
}

/// specialized wrapper for MPI_Bcast on receiver side
template <typename T, std::enable_if_t<!std::is_const<T>::value, int> = 0>
void receive_from(int broadcaster_rank, message<T>&& message, Communicator communicator) {
  MPI_Bcast(static_cast<void*>(message.ptr()), message.count(), message.mpi_type(), communicator.rank(),
            communicator);
}

}

namespace async_broadcast {

/// specialized wrapper for MPI_Ibcast on sender side
template <typename T>
void send(message<T>&& message, Communicator communicator,
          std::function<void()> action_before_retrying) {
  internal::async_bcast(communicator.rank(),
                        const_cast<std::remove_const_t<typename dlaf::comm::message<T>::T>*>(
                            message.ptr()),
                        message.count(), message.mpi_type(), communicator, action_before_retrying);
}

/// specialized wrapper for MPI_Ibcast on receiver side
template <typename T>
void receive_from(int broadcaster_rank, const message<T>& message, Communicator communicator,
                  std::function<void()> action_before_retrying) {
  internal::async_bcast(broadcaster_rank, message.ptr(), message.count(), message.mpi_type(),
                        communicator, action_before_retrying);
}

}

}
}
