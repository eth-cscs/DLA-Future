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
#include <mutex>

#include <mpi.h>
#include <hpx/mutex.hpp>

#include "dlaf/communication/error.h"

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

/// Serializes the MPI call if `MPI_THREAD_SERIALIZED` is used.
///
/// Note: HPX mutex is used instead of std::mutex to avoid stalling the HPX scheduler as the function may
///       be called from a HPX thread/task.
/// Note: This function should only be called after HPX has been initialized, otherwise the HPX mutex may
///       error out.
template <class F, class... Ts>
void mpi_invoke(F f, Ts... ts) noexcept {
  if (mpi_serialized()) {
    static hpx::lcos::local::mutex mt;
    std::lock_guard<hpx::lcos::local::mutex> lk(mt);
    DLAF_MPI_CALL(f(ts...));
  }
  else {
    DLAF_MPI_CALL(f(ts...));
  }
}

struct mpi_init {
  /// Initialize MPI to either MPI_THREAD_SERIALIZED or MPI_THREAD_MULTIPLE
  mpi_init(int argc, char** argv, mpi_thread_level thd_level) noexcept {
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
  ~mpi_init() noexcept {
    MPI_Finalize();
  }
};

}
}
