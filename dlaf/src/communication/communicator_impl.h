#pragma once

#include <mpi.h>

namespace dlaf {
namespace comm {

class Communicator;

constexpr bool is_manageable(MPI_Comm mpi_communicator) noexcept;

class CommunicatorImpl {
  friend class Communicator;

public:
  CommunicatorImpl(MPI_Comm mpi_communicator) noexcept(false);
  virtual ~CommunicatorImpl() = default;

  CommunicatorImpl(const CommunicatorImpl&) = delete;
  CommunicatorImpl& operator=(const CommunicatorImpl&) = delete;

protected:
  CommunicatorImpl() noexcept = default;

  void release();

  MPI_Comm comm_ = MPI_COMM_NULL;
  int size_ = 0;
  int rank_ = MPI_UNDEFINED;
};

class CommunicatorImpl_NotManaged : public CommunicatorImpl {
public:
  using CommunicatorImpl::CommunicatorImpl;
};

class CommunicatorImpl_Managed : public CommunicatorImpl {
public:
  CommunicatorImpl_Managed(MPI_Comm mpi_communicator) noexcept(false);
  ~CommunicatorImpl_Managed();
};

}
}
