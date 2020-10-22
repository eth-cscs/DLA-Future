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
#include "dlaf/common/data_descriptor.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"
#include "dlaf/communication/sync/broadcast.h"
#include "dlaf/communication/sync/reduce.h"

namespace dlaf {
namespace comm {
namespace sync {

/// MPI_AllReduce wrapper.
///
/// MPI AllReduce(see MPI documentation for additional info).
/// @param reduce_operation MPI_Op to perform on @p input data coming from ranks in @p communicator,
template <class DataIn, class DataOut>
void all_reduce(Communicator& communicator, MPI_Op reduce_operation, const DataIn input,
                const DataOut output) {
  using T = std::remove_const_t<typename common::data_traits<DataIn>::element_t>;

  // Wayout for single rank communicator, just copy data
  if (communicator.size() == 1) {
    common::copy(input, output);
    return;
  }

  // Data descriptors used internally, initialized with Data given as parameters,
  // but they may be replaced internally by contiguous Buffers in case of need
  common::DataDescriptor<const T> internal_input = input;
  common::DataDescriptor<T> internal_output = output;

  // Buffers not allocated, just placeholders in case we need to allocate them
  common::Buffer<T> temporary_buffer_in;
  common::Buffer<T> temporary_buffer_out;

  // if input is not contiguous, copy it in a contiguous temporary buffer
  if (!input.is_contiguous()) {
    // allocate the temporary buffer
    temporary_buffer_in = common::create_temporary_buffer(internal_input);
    // set it as internal intermediate input
    internal_input = temporary_buffer_in;
    // copy the data to the internal intermediate buffer
    common::copy(input, temporary_buffer_in);
  }

  // if output is not contiguous, create an intermediate buffer
  if (!output.is_contiguous()) {
    // allocate the temporary buffer
    temporary_buffer_out = common::create_temporary_buffer(internal_output);
    // and set it as internal intermediate output
    internal_output = temporary_buffer_out;
  }

  auto message_input = comm::make_message(std::move(internal_input));
  auto message_output = comm::make_message(DataOut(internal_output));

  MPI_Allreduce(message_input.data(), message_output.data(), message_input.count(),
                message_input.mpi_type(), reduce_operation, communicator);

  // if output was not contiguous, copy it back!
  if (!output.is_contiguous())
    common::copy(internal_output, output);
}

}
}
}
