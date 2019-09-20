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

class CommunicatorImpl_NotManaged;
class CommunicatorImpl_Managed;

Communicator::Communicator() : comm_ref_(new CommunicatorImpl()) {}

Communicator::Communicator(MPI_Comm mpi_communicator)
    : comm_ref_(new CommunicatorImpl(mpi_communicator)) {}

Communicator::Communicator(MPI_Comm mpi_communicator, managed) noexcept(false)
    : comm_ref_(new CommunicatorImpl_Managed(mpi_communicator)) {}

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
