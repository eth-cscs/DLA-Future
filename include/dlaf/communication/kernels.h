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

#include "dlaf/common/assert.h"
#include "dlaf/common/data.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/message.h"

namespace dlaf {
namespace comm {

/// MPI_Isend wrapper
///
/// For more information, see the Data concept in "dlaf/common/data.h"
template <class DataIn>
hpx::future<void> send(executor& ex, int receiver_rank, const DataIn& data) {
  int tag = 0;
  auto message = make_message(common::make_data(data));
  return ex.async_execute(MPI_Isend, message.data(), message.count(), message.mpi_type(), receiver_rank,
                          tag);
}

/// MPI_Irecv wrapper
///
/// For more information, see the Data concept in "dlaf/common/data.h"
template <class DataOut>
hpx::future<void> recv(executor& ex, int sender_rank, DataOut& data) {
  int tag = 0;
  auto message = make_message(common::make_data(data));
  return ex.async_execute(MPI_Irecv, message.data(), message.count(), message.mpi_type(), sender_rank,
                          tag);
}

/// MPI_Ibcast wrapper
///
/// For more information, see the Data concept in "dlaf/common/data.h"
template <class DataIn>
hpx::future<void> bcast(executor& ex, int root_rank, DataIn& tile) {
  auto data = common::make_data(tile);
  using DataT = std::remove_const_t<typename common::data_traits<decltype(data)>::element_t>;

  auto message = comm::make_message(common::make_data(data));
  return ex.async_execute(MPI_Ibcast, const_cast<DataT*>(message.data()), message.count(),
                          message.mpi_type(), root_rank);
}

/// MPI_Ireduce wrapper
///
/// @param reduce_operation MPI_Op to perform on @p input data coming from ranks in @p communicator
template <class DataIn, class DataOut>
hpx::future<void> reduce(executor& ex, int root_rank, MPI_Op reduce_operation, const DataIn& input,
                         const DataOut& output) {
  DLAF_ASSERT(input.is_contiguous(), "Input data has to be contiguous!");
  if (ex.comm().rank() == root_rank)
    DLAF_ASSERT(output.is_contiguous(), "Output data has to be contiguous!");

  auto in_msg = comm::make_message(common::make_data(input));
  auto out_msg = comm::make_message(common::make_data(output));
  return ex.async_execute(MPI_Ireduce, in_msg.data(), out_msg.data(), in_msg.count(), in_msg.mpi_type(),
                          reduce_operation, root_rank);
}

}
}
