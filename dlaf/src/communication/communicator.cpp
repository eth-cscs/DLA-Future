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

#include "communicator_impl.h"

namespace dlaf {
namespace comm {

Communicator::Communicator() : Communicator(MPI_COMM_NULL) {}

Communicator::Communicator(MPI_Comm mpi_communicator)
   : comm_ref_(new CommunicatorImpl(mpi_communicator, false)) {}

Communicator::Communicator(MPI_Comm mpi_communicator, Managed) noexcept(false)
   : comm_ref_(new CommunicatorImpl(mpi_communicator, true)) {}

Communicator::operator MPI_Comm() const noexcept {
  return comm_ref_->comm_;
}

MPI_Comm* Communicator::operator&() noexcept {
  return &(comm_ref_->comm_);
}

int Communicator::rank() const noexcept {
  return comm_ref_->rank_;
}

int Communicator::size() const noexcept {
  return comm_ref_->size_;
}

}
}
