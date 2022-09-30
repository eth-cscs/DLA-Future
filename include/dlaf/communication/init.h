//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <iostream>
#include <mutex>

#include <mpi.h>
#include <pika/mutex.hpp>

#include "dlaf/communication/error.h"

namespace dlaf {
namespace comm {

struct mpi_init {
  /// Initialize MPI to MPI_THREAD_MULTIPLE
  mpi_init(int argc, char** argv) noexcept {
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
