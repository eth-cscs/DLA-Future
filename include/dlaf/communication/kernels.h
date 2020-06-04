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

#include <mpi.h>

#include "dlaf/common/data.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/message.h"

namespace dlaf {
namespace comm {

/// MPI_Isend wrapper for sender side accepting a Data
///
/// For more information, see the Data concept in "dlaf/common/data.h"
template <class DataIn>
hpx::future<void> send(executor& ex, int receiver_rank, DataIn&& data) {
  int tag = 0;
  auto message = make_message(common::make_data(std::forward<DataIn>(data)));
  return ex.async_execute(MPI_Isend, message.data(), message.count(), message.mpi_type(), receiver_rank,
                          tag);
}

/// MPI_Irecv wrapper for receiver side accepting a Data
///
/// For more information, see the Data concept in "dlaf/common/data.h"
template <class DataOut>
hpx::future<void> recv(executor& ex, int sender_rank, DataOut&& data) {
  int tag = 0;
  auto message = make_message(common::make_data(std::forward<DataOut>(data)));
  return ex.async_execute(MPI_Irecv, message.data(), message.count(), message.mpi_type(), sender_rank,
                          tag);
}

/// MPI_Ibcast wrapper for sender side accepting a Data
///
/// For more information, see the Data concept in "dlaf/common/data.h"
template <class DataIn>
hpx::future<void> bcast(executor& ex, int root_rank, DataIn&& message_to_send) {
  auto data = common::make_data(message_to_send);
  using DataT = std::remove_const_t<typename common::data_traits<decltype(data)>::element_t>;
  auto message = comm::make_message(std::move(data));
  return ex.async_execute(MPI_Ibcast, const_cast<DataT*>(message.data()), message.count(),
                          message.mpi_type(), root_rank);
}

}
}
