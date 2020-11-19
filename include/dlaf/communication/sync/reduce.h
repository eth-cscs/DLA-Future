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

namespace dlaf {
namespace comm {
namespace sync {

namespace internal {
namespace reduce {

/// MPI_Reduce wrapper (sender side).
///
/// MPI Reduce(see MPI documentation for additional info).
/// @param reduce_operation MPI_Op to perform on @p input data coming from ranks in @p communicator.
template <class DataIn, class DataOut>
void collector(Communicator& communicator, MPI_Op reduce_operation, const DataIn input,
               const DataOut output) {
  using T = std::remove_const_t<typename common::data_traits<DataIn>::element_t>;

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
             message_input.mpi_type(), reduce_operation, communicator.rank(), communicator));

  // if the output buffer has been used, copy-back output values
  if (buffer_out)
    common::copy(buffer_out, output);
}

/// MPI_Reduce wrapper (receiver side).
///
/// MPI Reduce(see MPI documentation for additional info).
/// @param rank_root  the rank that will collect the result in output,
/// @param reduce_operation MPI_Op to perform on @p input data coming from ranks in @p communicator.
template <class DataIn>
void participant(int rank_root, Communicator& communicator, MPI_Op reduce_operation,
                 const DataIn input) {
  using T = std::remove_const_t<typename common::data_traits<DataIn>::element_t>;

  // Buffers not allocated, just placeholders in case we need to allocate them
  common::Buffer<T> buffer_in;

  auto message_input = comm::make_message(make_contiguous(input, buffer_in));

  // if the input buffer has been used, initialize it with input values
  if (buffer_in)
    common::copy(input, buffer_in);

  DLAF_MPI_CALL(MPI_Reduce(message_input.data(), nullptr, message_input.count(), message_input.mpi_type(),
             reduce_operation, rank_root, communicator));
}
}
}

/// MPI_Reduce wrapper.
///
/// MPI Reduce(see MPI documentation for additional info).
/// @param rank_root the rank that will collect the result in output,
/// @param reduce_operation MPI_Op to perform on @p input data coming from ranks in @p communicator,
/// @pre @p rank_root < @p communicator.size(),
/// @pre @p rank_root != MPI_UNDEFINED.
template <class DataIn, class DataOut>
void reduce(const int rank_root, Communicator& communicator, MPI_Op reduce_operation, const DataIn input,
            const DataOut output) {
  DLAF_ASSERT(rank_root < communicator.size() && rank_root != MPI_UNDEFINED, "The rank is not valid!",
              rank_root, communicator.size());

  if (rank_root == communicator.rank())
    internal::reduce::collector(communicator, reduce_operation, input, output);
  else
    internal::reduce::participant(rank_root, communicator, reduce_operation, input);
}
}
}
}
