#pragma once

/// @file

#include "mpi.h"

#include <cstdio>

namespace dlaf {
namespace comm {

/// A RAII type to simplify MPI's init/finalize
struct InitMPI {

  InitMPI(int argc, char** argv, int thd_required) {
    int thd_provided;
    MPI_Init_thread(&argc, &argv, thd_required, &thd_provided);

    if (thd_required != thd_provided) {
      std::printf("Provided MPI threading model does not match the required one.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  ~InitMPI() {
    MPI_Finalize();
  }
};

}
}
