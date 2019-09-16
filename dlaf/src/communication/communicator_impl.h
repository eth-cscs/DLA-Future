#pragma once

#include <mpi.h>

namespace dlaf {
namespace comm {

class Communicator;

constexpr bool is_manageable(MPI_Comm mpi_communicator) noexcept;

class CommunicatorImpl {
  friend class Communicator;

  public:
  CommunicatorImpl(MPI_Comm mpi_communicator, bool to_manage);
  ~CommunicatorImpl();

  protected:
  void release();

  bool is_managed_ = false;
  MPI_Comm comm_;
  int size_ = 0;
  int rank_ = MPI_UNDEFINED;
};

}
}
