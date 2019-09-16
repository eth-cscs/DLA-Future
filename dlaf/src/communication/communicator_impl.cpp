#include "communicator_impl.h"

#include <stdexcept>

namespace dlaf {
namespace comm {

constexpr bool is_manageable(MPI_Comm mpi_communicator) noexcept {
  switch(mpi_communicator) {
    case MPI_COMM_WORLD:
    case MPI_COMM_NULL:
      return false;
  }
  return true;
}

CommunicatorImpl::CommunicatorImpl(MPI_Comm mpi_communicator, bool to_manage)
    : comm_(mpi_communicator) {
  if (to_manage) {
    if (!is_manageable(mpi_communicator))
      throw std::runtime_error("MPI_COMM_WORLD and MPI_COMM_NULL can not be managed");
    else
      is_managed_ = true;
  }

  if (MPI_COMM_NULL != mpi_communicator) {
    MPI_Comm_size(comm_, &size_);
    MPI_Comm_rank(comm_, &rank_);
  }
}

CommunicatorImpl::~CommunicatorImpl() {
  if (is_managed_)
    release();
}

void CommunicatorImpl::release() {
  if (is_managed_)
    MPI_Comm_free(&comm_);
}

}
}
