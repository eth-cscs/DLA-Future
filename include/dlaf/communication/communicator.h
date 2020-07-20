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
#include "dlaf/communication/error.h"

namespace dlaf {
namespace comm {

class CommunicatorImpl;

/// MPI-compatible wrapper for the MPI_Comm.
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
  friend Communicator make_communicator_managed(MPI_Comm);

public:
  /// Create a NULL Communicator (i.e. MPI_COMM_NULL).
  Communicator();

  /// Wrap an MPI_Comm into a Communicator.
  ///
  /// User keeps the ownership of the MPI_Comm, but this object has the usage exclusiveness.
  /// The user has to grant this, otherwise it leads to UB.
  /// @param mpi_communicator MPI_Comm to wrap.
  Communicator(MPI_Comm mpi_communicator);

  /// Return the internal MPI_Comm handler.
  ///
  /// Useful for MPI function calls.
  operator MPI_Comm() const noexcept;

  /// Return the pointer to the internal MPI_Comm handler.
  ///
  /// Useful for MPI function calls.
  MPI_Comm* operator&() noexcept;
  const MPI_Comm* operator&() const noexcept;

  /// Return the rank of the current process in the Communicator.
  int rank() const noexcept;
  /// Return the number of ranks in the Communicator.
  int size() const noexcept;

private:
  /// Tag to give to constructor in order to give MPI_Comm ownership to Communicator.
  struct managed {};

  ///  Wrap and manage an MPI_Comm into a Communicator.
  ///
  /// This object takes the ownership of the MPI_Comm.
  /// @param mpi_communicator MPI_Comm to wrap,
  /// @param managed tag (anonymous parameter).
  Communicator(MPI_Comm mpi_communicator, managed);

  std::shared_ptr<CommunicatorImpl> comm_ref_;
};

/// Wrap an MPI_Comm into a Communicator that takes the ownership.
inline Communicator make_communicator_managed(MPI_Comm mpi_communicator) {
  return Communicator(mpi_communicator, Communicator::managed{});
}

}
}
