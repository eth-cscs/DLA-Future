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

#include "dlaf/common/data.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"

namespace dlaf {
namespace comm {
namespace sync {

/// MPI_Send wrapper for sender side accepting a Data
///
/// For more information, see the Data concept in "dlaf/common/data.h"
template <class DataIn>
void send_to(int receiver_rank, Communicator& communicator, DataIn&& data) {
  int tag = 0;
  auto message = make_message(common::make_data(std::forward<DataIn>(data)));
  MPI_Send(message.data(), message.count(), message.mpi_type(), receiver_rank, tag, communicator);
}

/// MPI_Recv wrapper for receiver side accepting a Data
///
/// For more information, see the Data concept in "dlaf/common/data.h"
template <class DataOut>
void receive_from(int sender_rank, Communicator& communicator, DataOut&& data) {
  int tag = 0;
  auto message = make_message(common::make_data(std::forward<DataOut>(data)));
  MPI_Recv(message.data(), message.count(), message.mpi_type(), sender_rank, tag, communicator,
           MPI_STATUS_IGNORE);
}
}
}
}
