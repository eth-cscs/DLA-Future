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
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"

namespace dlaf {
namespace comm {
namespace sync {

/// MPI_Reduce wrapper
/// MPI Reduce(see MPI documentation for additional info)
/// @param rank_root  the rank that will collect the result in output
/// @param reduce_operation MPI_Op to perform on @p input data coming from ranks in @p communicator
template <class T>
void reduce(int rank_root, Communicator& communicator, MPI_Op reduce_operation, Message<T>&& input,
            Message<T>&& output) {
  MPI_Reduce(input.data(), output.data(), input.count(), input.mpi_type(), reduce_operation, rank_root,
             communicator);
}

}
}
}
