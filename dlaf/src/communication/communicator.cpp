//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/communication/communicator.h"

namespace dlaf {
namespace comm {

Communicator::Communicator()
: Communicator(MPI_COMM_NULL) {}

Communicator::Communicator(MPI_Comm mpi_communicator)
: comm_(mpi_communicator) {
  if (MPI_COMM_NULL != mpi_communicator) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
  }
  else {
    rank_ = MPI_UNDEFINED;
    size_ = 0;
  }
}

Communicator::operator MPI_Comm() const noexcept { return comm_; }
int Communicator::rank() const noexcept { return rank_; }
int Communicator::size() const noexcept { return size_; }
void Communicator::release() { MPI_CALL(MPI_Comm_free(&comm_)); }

}
}
