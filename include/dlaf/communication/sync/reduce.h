//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
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

namespace dlaf {
namespace comm {
namespace sync {

/// MPI_Reduce wrapper (collector side, i.e. send and receive).
///
/// MPI Reduce(see MPI documentation for additional info).
/// @param reduce_op MPI_Op to perform on @p input data coming from ranks in @p communicator.
template <class DataIn, class DataOut>
void reduce(Communicator& communicator, MPI_Op reduce_op, const DataIn input, const DataOut output) {
  using T = std::remove_const_t<typename common::data_traits<DataIn>::element_t>;

  DLAF_ASSERT_MODERATE(input != output, "input and output should not equal (use in-place)");

  // Wayout for single rank communicator, just copy data
  if (communicator.size() == 1) {
    common::copy(input, output);
    return;
  }

  // Buffers not allocated, just placeholders in case we need to allocate them
  common::Buffer<T> buffer_in, buffer_out;

  auto message_input = comm::make_message(make_contiguous(input, buffer_in));
  auto message_output = comm::make_message(make_contiguous(output, buffer_out));

  // if the input buffer has been used, initialize it with input values
  if (buffer_in)
    common::copy(input, buffer_in);

  DLAF_MPI_CALL(MPI_Reduce(message_input.data(), message_output.data(), message_input.count(),
                           message_input.mpi_type(), reduce_op, communicator.rank(), communicator));

  // if the output buffer has been used, copy-back output values
  if (buffer_out)
    common::copy(buffer_out, output);
}

/// MPI_Reduce wrapper (collector-side, in-place).
///
/// MPI Reduce(see MPI documentation for additional info).
/// It uses the MPI_IN_PLACE option, so the result of the reduce is stored in the same buffer
/// used for the input, i.e. @p inout
///
/// It must be highlighted that if the buffer is not contiguous, it will get copied to a support
/// buffer, and then copied back. From the user perspective it is still in-place, but in that specific
/// case there will be internal memory copies.
///
/// @param reduce_op MPI_Op to perform on @p inout data coming from ranks in @p communicator,
/// @pre @p rank_root < @p communicator.size(),
template <class DataInOut>
void reduce(Communicator& communicator, MPI_Op reduce_op, const DataInOut inout) {
  using T = std::remove_const_t<typename common::data_traits<DataInOut>::element_t>;

  // Wayout for single rank communicator, just copy data
  if (communicator.size() == 1)
    return;

  // Buffer not allocated, just a placeholder in case we need to allocate it
  common::Buffer<T> buffer_inout;

  auto message_inout = comm::make_message(make_contiguous(inout, buffer_inout));

  // if the buffer has been used, initialize it with input values
  if (buffer_inout)
    common::copy(inout, buffer_inout);

  DLAF_MPI_CALL(MPI_Reduce(MPI_IN_PLACE, message_inout.data(), message_inout.count(),
                           message_inout.mpi_type(), reduce_op, communicator.rank(), communicator));

  // if the buffer has been used, copy-back output values
  if (buffer_inout)
    common::copy(buffer_inout, inout);
}

/// MPI_Reduce wrapper (sender side).
///
/// MPI Reduce(see MPI documentation for additional info).
///
/// This is an helper function for the sender side, with just needed arguments (i.e. output parameter
/// is meaningful just on the sender side).
/// @param rank_root  the rank that will collect the result in output,
/// @param reduce_op MPI_Op to perform on @p input data coming from ranks in @p communicator.
template <class DataIn>
void reduce(IndexT_MPI rank_root, Communicator& communicator, MPI_Op reduce_op, const DataIn input) {
  using T = std::remove_const_t<typename common::data_traits<DataIn>::element_t>;

  // Buffers not allocated, just placeholders in case we need to allocate them
  common::Buffer<T> buffer_in;

  auto message_input = comm::make_message(make_contiguous(input, buffer_in));

  // if the input buffer has been used, initialize it with input values
  if (buffer_in)
    common::copy(input, buffer_in);

  DLAF_MPI_CALL(MPI_Reduce(message_input.data(), nullptr, message_input.count(),
                           message_input.mpi_type(), reduce_op, rank_root, communicator));
}

/// MPI_Reduce wrapper (both sides).
///
/// MPI Reduce(see MPI documentation for additional info).
/// @param rank_root the rank that will collect the result in output,
/// @param reduce_op MPI_Op to perform on @p input data coming from ranks in @p communicator,
/// @pre @p 0 <= rank_root < @p communicator.size(),
/// @pre @p rank_root != MPI_UNDEFINED.
template <class DataIn, class DataOut>
void reduce(const IndexT_MPI rank_root, Communicator& communicator, MPI_Op reduce_op, const DataIn input,
            const DataOut output) {
  DLAF_ASSERT(0 <= rank_root && rank_root < communicator.size() && rank_root != MPI_UNDEFINED, rank_root,
              communicator.size());

  if (rank_root == communicator.rank())
    reduce(communicator, reduce_op, input, output);
  else
    reduce(rank_root, communicator, reduce_op, input);
}
}
}
}
