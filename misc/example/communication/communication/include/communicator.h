#pragma once

#include "mpi_header.h"

namespace dlaf
{
namespace comm
{

/// This class is a wrapper for MPI_Comm, it does not get ownership of it
class Communicator {
  public:
  Communicator() noexcept(false);
  Communicator(MPI_Comm mpi_communicator) noexcept(false);

  explicit operator MPI_Comm() const noexcept;

  int rank() const noexcept;
  int size() const noexcept;

  protected:
  void release() noexcept(false);

  private:
  MPI_Comm comm_;
  int rank_;
  int size_;
};

}
}
