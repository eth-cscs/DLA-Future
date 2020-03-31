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

#include "dlaf/common/assert.h"

#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"

#include "dlaf/common/data.h"

namespace dlaf {
namespace comm {
namespace sync {
namespace broadcast {

/// MPI_Bcast wrapper for sender side accepting a Data
///
/// For more information, see the Data concept in "dlaf/common/data.h"
template <class DataIn>
void send(Communicator& communicator, DataIn&& message_to_send) {
  auto data = common::make_data(message_to_send);
  using DataT = std::remove_const_t<typename common::data_traits<decltype(data)>::element_t>;

  auto message = comm::make_message(std::move(data));
  MPI_Bcast(const_cast<DataT*>(message.data()), message.count(), message.mpi_type(), communicator.rank(),
            communicator);
}

/// MPI_Bcast wrapper for receiver side accepting a dlaf::comm::Message
///
/// For more information, see the Data concept in "dlaf/common/data.h"
template <class DataOut>
void receive_from(const int broadcaster_rank, Communicator& communicator, DataOut&& data) {
  DLAF_ASSERT_HEAVY((broadcaster_rank != communicator.rank()));
  auto message = comm::make_message(common::make_data(std::forward<DataOut>(data)));
  MPI_Bcast(message.data(), message.count(), message.mpi_type(), broadcaster_rank, communicator);
}
}
}
}
}
