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

/// @brief MPI-compatible wrapper for the MPI_Comm.
///
/// Being MPI-compatible means that it can be used in MPI calls, in fact, it is implicitly converted
/// to MPI_Comm and the reference of a Communicator instance returns the pointer to the internal MPI_Comm.
/// In case of need, a pointer to Communicator instance can be obtained using std::addressof.
///
/// MPI_Comm validity must be granted by the user. A copy of a Communicator refers exactly to the same
/// MPI_Comm of the original one (i.e. MPI_Comm is not duplicated).
class Communicator {
  public:
  /// @brief Create a NULL Communicator (i.e. MPI_COMM_NULL)
  Communicator() noexcept(false);

  /// @brief Wrap an MPI_Comm into a Communicator
  ///
  /// The validity of the wrapped MPI_Comm must be granted and managed by the user, otherwise an UB occurs.
  ///
  /// @p mpi_communicator MPI_Comm to wrap.
  Communicator(MPI_Comm mpi_communicator) noexcept(false);

  /// @brief Return the internal MPI_Comm handler
  ///
  /// Useful for MPI function calls
  operator MPI_Comm() const noexcept;

  /// @brief Return the pointer to the internal MPI_Comm handler
  ///
  /// Useful for MPI function calls
  MPI_Comm * operator &() noexcept;

  /// @brief Return the rank of the current process in the Communicator
  int rank() const noexcept;
  /// @brief Return the number of ranks in the Communicator
  int size() const noexcept;

  protected:
  /// @brief Release the underlying communicator
  ///
  /// It calls MPI_Comm_free (for additional info see MPI documentation)
  void release() noexcept(false);

  private:
  MPI_Comm comm_;
  int rank_ = MPI_UNDEFINED;
  int size_ = 0;
};

}
}
