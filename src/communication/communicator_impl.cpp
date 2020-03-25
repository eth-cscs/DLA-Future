#include "communicator_impl.h"

#include <cassert>
#include <stdexcept>

#include "dlaf/mpi_header.h"

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
  assert(comm_ != MPI_COMM_NULL);
  MPI_CALL(MPI_Comm_size(comm_, &size_));
  MPI_CALL(MPI_Comm_rank(comm_, &rank_));
}

CommunicatorImpl_Managed::CommunicatorImpl_Managed(MPI_Comm mpi_communicator)
    : CommunicatorImpl(mpi_communicator) {
  if (!is_manageable(comm_))
    throw std::invalid_argument("Passed communicator is not manageable");
}

CommunicatorImpl_Managed::~CommunicatorImpl_Managed() {
  MPI_CALL(MPI_Comm_free(&comm_));
}

}
}
