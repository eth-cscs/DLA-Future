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

#include <iostream>

#include <mpi.h>

namespace dlaf {
namespace comm {

enum class mpi_thread_level { serialized, multiple };

/// Checks if MPI was initialized with MPI_THREAD_SERIALIZED.
inline bool mpi_serialized() noexcept {
  // This is executed only once and is thread safe.
  // Note that the lambda is called at the end.
  static bool is_serialized = []() {
    int provided;
    MPI_Query_thread(&provided);
    if (provided == MPI_THREAD_SERIALIZED)
      return true;
    else if (provided == MPI_THREAD_MULTIPLE)
      return false;
    else {
      std::cerr << "MPI must be initialized to either `MPI_THREAD_SERIALIZED` or `MPI_THREAD_MULTIPLE`!";
      MPI_Abort(MPI_COMM_WORLD, 1);
      return false;  // unreachable - here to avoid compiler warnings
    }
  }();
  return is_serialized;
}

/// Initialize MPI to either MPI_THREAD_SERIALIZED or MPI_THREAD_MULTIPLE
inline void mpi_init(int argc, char** argv, mpi_thread_level thd_level) noexcept {
  int required_threading;
  if (thd_level == mpi_thread_level::serialized)
    required_threading = MPI_THREAD_SERIALIZED;
  else
    required_threading = MPI_THREAD_MULTIPLE;

  int provided_threading;
  MPI_Init_thread(&argc, &argv, required_threading, &provided_threading);

  if (provided_threading != required_threading) {
    std::cerr << "Provided MPI threading model does not match the required one." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

/// Finalize MPI
inline void mpi_fin() noexcept {
  MPI_Finalize();
}

}
}
