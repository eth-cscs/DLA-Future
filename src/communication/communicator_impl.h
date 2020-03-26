#pragma once

#include "dlaf/mpi_header.h"

/// @file

namespace dlaf {
namespace comm {

class Communicator;

/// @return true if it is a manageable communicator (i.e. it is not MPI_COMM_NULL or MPI_COMM_WORLD)
bool is_manageable(MPI_Comm mpi_communicator) noexcept;

/// Basic wrapper for MPI_Comm
/// It is more or less an alias for MPI_Comm
class CommunicatorImpl {
  friend class Communicator;

public:
  /// Create a wrapper for the given MPI_Comm
  /// @pre mpi_communicator != MPI_COMM_NULL
  CommunicatorImpl(MPI_Comm mpi_communicator);

  /// It does nothing, it is just needed for extending destruction behavior with inheritance
  virtual ~CommunicatorImpl() = default;

  CommunicatorImpl(const CommunicatorImpl&) = delete;
  CommunicatorImpl& operator=(const CommunicatorImpl&) = delete;

protected:
  /// It creates a wrapper for a NULL communicator
  CommunicatorImpl() noexcept = default;

  MPI_Comm comm_ = MPI_COMM_NULL;
  int size_ = 0;
  int rank_ = MPI_UNDEFINED;
};

/// MPI_Comm wrapper releasing MPI resources on destruction
class CommunicatorImpl_Managed : public CommunicatorImpl {
public:
  /// @pre @p mpi_communicator must be a manageable communicator (see dlaf::comm::is_manageable())
  CommunicatorImpl_Managed(MPI_Comm mpi_communicator);

  /// It calls MPI_Comm_free (for additional info, see MPI documentation)
  ~CommunicatorImpl_Managed();
};

}
}
