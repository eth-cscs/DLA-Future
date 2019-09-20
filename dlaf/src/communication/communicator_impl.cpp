#include "communicator_impl.h"

#include <cassert>
#include <stdexcept>

#include "dlaf/mpi_header.h"

namespace dlaf {
namespace comm {

constexpr bool is_manageable(MPI_Comm mpi_communicator) noexcept {
  switch (mpi_communicator) {
    case mpi::WORLD_COMMUNICATOR:
    case mpi::NULL_COMMUNICATOR:
      return false;
  }
  return true;
}

CommunicatorImpl::CommunicatorImpl(MPI_Comm mpi_communicator) : comm_(mpi_communicator) {
  assert(comm_ != MPI_COMM_NULL);
  MPI_CALL(MPI_Comm_size(comm_, &size_));
  MPI_CALL(MPI_Comm_rank(comm_, &rank_));
}

void CommunicatorImpl::release() {
  MPI_CALL(MPI_Comm_free(&comm_));
}

CommunicatorImpl_Managed::CommunicatorImpl_Managed(MPI_Comm mpi_communicator) noexcept(false)
    : CommunicatorImpl(mpi_communicator) {
  if (!is_manageable(comm_))
    throw std::invalid_argument("Passed communicator is not manageable");
}

CommunicatorImpl_Managed::~CommunicatorImpl_Managed() {
  release();
}

}
}
