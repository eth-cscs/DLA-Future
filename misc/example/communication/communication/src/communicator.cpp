#include "communicator.h"

#include <iostream>

namespace dlaf {
namespace comm {

Communicator::Communicator()
: Communicator(mpi::COMM_WORLD) {}

Communicator::Communicator(MPI_Comm mpi_communicator)
: comm_(mpi_communicator) {
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &size_);
}

Communicator::operator MPI_Comm() const noexcept { return comm_; }
int Communicator::rank() const noexcept { return rank_; }
int Communicator::size() const noexcept { return size_; }
void Communicator::release() { MPI_CALL(MPI_Comm_free(&comm_)); }

}
}
