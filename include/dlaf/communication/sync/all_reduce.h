//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <chrono>

#include <mpi.h>

#include <pika/execution.hpp>

#include <dlaf/common/assert.h>
#include <dlaf/common/data_descriptor.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/message.h>
#include <dlaf/communication/sync/reduce.h>
#include <dlaf/sender/transform_mpi.h>

namespace dlaf {
namespace comm {
namespace sync {

/// MPI_AllReduce wrapper.
///
/// MPI AllReduce(see MPI documentation for additional info).
/// @param reduce_operation MPI_Op to perform on @p input data coming from ranks in @p communicator,
template <class DataIn, class DataOut>
void allReduce(Communicator& communicator, MPI_Op reduce_operation, const DataIn input,
               const DataOut output) {
  using common::make_contiguous;

  using T = std::remove_const_t<typename common::data_traits<DataIn>::element_t>;

  // Wayout for single rank communicator, just copy data
  if (communicator.size() == 1) {
    DLAF_ASSERT_MODERATE(input != output, "input and output should not equal (use in-place)");
    common::copy(input, output);
    return;
  }

  // Buffers not allocated, just placeholders in case we need to allocate them
  common::Buffer<T> buffer_in, buffer_out;

  auto message_input = comm::make_message(make_contiguous(input, buffer_in));
  auto message_output = comm::make_message(make_contiguous(output, buffer_out));

  DLAF_ASSERT_MODERATE((buffer_in || buffer_out) || (input != output),
                       "input and output should not equal (use in-place)");

  // if the input buffer has been used, initialize it with input values
  if (buffer_in)
    common::copy(input, buffer_in);

  DLAF_MPI_CHECK_ERROR(MPI_Allreduce(message_input.data(), message_output.data(), message_input.count(),
                                     message_input.mpi_type(), reduce_operation, communicator));

  // if the output buffer has been used, copy-back output values
  if (buffer_out)
    common::copy(buffer_out, output);
}

/// MPI_AllReduce wrapper (in-place)
///
/// MPI AllReduce(see MPI documentation for additional info).
/// @param reduce_operation MPI_Op to perform on @p input data coming from ranks in @p communicator,
template <class DataInOut>
void allReduceInPlace(Communicator& communicator, MPI_Op reduce_operation, const DataInOut inout) {
  using common::make_contiguous;

  using T = std::remove_const_t<typename common::data_traits<DataInOut>::element_t>;

  // Wayout for single rank communicator, just copy data
  if (communicator.size() == 1)
    return;

  // Buffers not allocated, just placeholders in case we need to allocate them
  common::Buffer<T> buffer_inout;

  auto message_inout = comm::make_message(make_contiguous(inout, buffer_inout));

  // if the input buffer has been used, initialize it with input values
  if (buffer_inout)
    common::copy(inout, buffer_inout);

#ifdef DLAF_ALLREDUCE_SYNC_WAIT
#ifndef DLAF_ALLREDUCE_SYNC_WAIT_BUSY_WAIT_TIME_US
#define DLAF_ALLREDUCE_SYNC_WAIT_BUSY_WAIT_TIME_US 0
#endif
  using dlaf::comm::internal::transformMPI;
  using dlaf::internal::whenAllLift;
  using pika::this_thread::experimental::sync_wait;
  sync_wait(whenAllLift(std::cref(communicator), reduce_operation, std::move(message_inout)) |
            transformMPI([](const Communicator& communicator, MPI_Op reduce_operation, auto&& msg, MPI_Request* req) {
              DLAF_MPI_CHECK_ERROR(MPI_Iallreduce(MPI_IN_PLACE, msg.data(), msg.count(), msg.mpi_type(),
                                                  reduce_operation, communicator, req));
            }), std::chrono::microseconds(DLAF_ALLREDUCE_SYNC_WAIT_BUSY_WAIT_TIME_US));
#else
  DLAF_MPI_CHECK_ERROR(MPI_Allreduce(MPI_IN_PLACE, message_inout.data(), message_inout.count(),
                                     message_inout.mpi_type(), reduce_operation, communicator));
#endif

  // if the output buffer has been used, copy-back output values
  if (buffer_inout)
    common::copy(buffer_inout, inout);
}
}
}
}
