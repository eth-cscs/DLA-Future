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

namespace broadcast {

/// specialized wrapper for MPI_Bcast on sender side
template <class T>
void send(Message<T>&& message, Communicator& communicator) {
  MPI_Bcast(const_cast<std::remove_const_t<typename dlaf::comm::Message<T>::element_t>*>(message.data()),
            message.count(), message.mpi_type(), communicator.rank(), communicator);
}

/// specialized wrapper for MPI_Bcast on receiver side
template <typename T, std::enable_if_t<!std::is_const<T>::value, int> = 0>
void receive_from(int broadcaster_rank, Message<T>&& message, Communicator communicator) {
  MPI_Bcast(message.data(), message.count(), message.mpi_type(), communicator.rank(), communicator);
}

}

}
}
