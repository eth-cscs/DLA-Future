//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "dlaf/mpi_header.h"

namespace dlaf {
namespace comm {

/// @brief A wrapper for the MPI_Comm handler.
///
/// MPI_Comm validity must be granted by the user
/// @pre MPI_Comm must be a valid handler to the communicator
/// @post MPI_Comm must still be a valid handler to the same communicator
class Communicator {
  public:
  /// @brief Create a NULL Communicator (i.e. MPI_COMM_NULL)
  Communicator() noexcept(false);

  /// @brief Wrap an MPI_Comm into a Communicator
  ///
  /// @p mpi_communicator MPI_Comm to wrap. Its validity must be granted by the user during the entire
  /// lifetime of the Communicator object, otherwise UB occurs. It is neither released on destruction.
  Communicator(MPI_Comm mpi_communicator) noexcept(false);

  /// Cast to MPI_Comm to be used in MPI calls
  explicit operator MPI_Comm() const noexcept;

  /// @brief Return the rank in the Communicator
  int rank() const noexcept;
  /// @brief Return the size of the Communicator
  int size() const noexcept;

  protected:
  /// @brief Release the underlying communicator
  ///
  /// It calls MPI_Comm_release for the wrapped MPI_Comm
  void release() noexcept(false);

  private:
  MPI_Comm comm_;
  int rank_ = MPI_UNDEFINED;
  int size_ = MPI_UNDEFINED;
};

}
}
