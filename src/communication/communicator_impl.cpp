//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "communicator_impl.h"

#include "dlaf/common/assert.h"
#include "dlaf/communication/error.h"

namespace dlaf {
namespace comm {

bool is_manageable(MPI_Comm mpi_communicator) noexcept {
  if (mpi_communicator == MPI_COMM_WORLD)
    return false;
  else if (mpi_communicator == MPI_COMM_NULL)
    return false;

  return true;
}

CommunicatorImpl::CommunicatorImpl(MPI_Comm mpi_communicator) : comm_(mpi_communicator) {
  DLAF_ASSERT(comm_ != MPI_COMM_NULL, "");
  DLAF_MPI_CALL(MPI_Comm_size(comm_, &size_));
  DLAF_MPI_CALL(MPI_Comm_rank(comm_, &rank_));
}

CommunicatorImpl_Managed::CommunicatorImpl_Managed(MPI_Comm mpi_communicator)
    : CommunicatorImpl(mpi_communicator) {
  DLAF_ASSERT(is_manageable(comm_), "");
}

CommunicatorImpl_Managed::~CommunicatorImpl_Managed() {
  DLAF_MPI_CALL(MPI_Comm_free(&comm_));
}

}
}
