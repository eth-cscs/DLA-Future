//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

/// @file

#include <iostream>
#include <mutex>

#include <mpi.h>

#ifdef DLAF_WITH_HIP
#include <hip/hip_runtime.h>
#include <whip.hpp>
#endif

#include <pika/mutex.hpp>

#include <dlaf/communication/error.h>

namespace dlaf {
namespace comm {

struct mpi_init {
  /// Initialize MPI to MPI_THREAD_MULTIPLE
  mpi_init(int argc, char** argv) noexcept {
    // On older Cray MPICH versions initializing HIP after MPI leads to HIP not seeing any devices. Hence
    // we eagerly initialize HIP here before MPI.
#ifdef DLAF_WITH_HIP
    whip::check_error(hipInit(0));
#endif

    int required_threading = MPI_THREAD_MULTIPLE;
    int provided_threading;
    MPI_Init_thread(&argc, &argv, required_threading, &provided_threading);

    if (provided_threading != required_threading) {
      std::cerr << "Provided MPI threading model does not match the required one (MPI_THREAD_MULTIPLE)."
                << std::endl;
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
