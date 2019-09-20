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

#include <memory>
#include "dlaf/mpi_header.h"

namespace dlaf {
namespace comm {

class CommunicatorImpl;

/// @brief MPI-compatible wrapper for the MPI_Comm.
///
/// Being MPI-compatible means that it can be used in MPI calls, in fact, it is implicitly converted
/// to MPI_Comm and the reference of a Communicator instance returns the pointer to the internal
/// MPI_Comm. In case of need, a pointer to Communicator instance can be obtained using std::addressof.
///
/// Apart from the default constructor Communicator() that creates a NULL communicator, the other two
/// constructors differ on MPI_Comm management: Communicator(MPI_Comm) leaves the ownership to the user,
/// that will have to grant its validity for the entire Communicator lifetime; instead,
/// Communicator(MPI_Comm, managed) acquires the ownership and will release it on destroy.
///
/// A copy of a Communicator refers exactly to the same MPI_Comm of the original one (i.e. MPI_Comm is
/// not duplicated).
class Communicator {
public:
  /// Tag to give to constructor in order to give MPI_Comm ownership to Communicator
  struct managed {};

  /// @brief Create a NULL Communicator (i.e. MPI_COMM_NULL)
  Communicator() noexcept(false);

  /// @brief Wrap an MPI_Comm into a Communicator
  ///
  /// The validity of the wrapped MPI_Comm must be granted and managed by the user, otherwise an UB occurs.
  /// @param mpi_communicator MPI_Comm to wrap.
  Communicator(MPI_Comm mpi_communicator) noexcept(false);

  /// @brief Wrap and manage an MPI_Comm into a Communicator
  ///
  /// The management of the underlying MPI_Comm is up to this Communicator and not to the user
  /// @param mpi_communicator MPI_Comm to wrap.
  Communicator(MPI_Comm mpi_communicator, managed) noexcept(false);

  /// @brief Return the internal MPI_Comm handler
  ///
  /// Useful for MPI function calls
  operator MPI_Comm() const noexcept;

  /// @brief Return the pointer to the internal MPI_Comm handler
  ///
  /// Useful for MPI function calls
  MPI_Comm* operator&() noexcept;

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
  std::shared_ptr<CommunicatorImpl> comm_ref_;
};

}
}
