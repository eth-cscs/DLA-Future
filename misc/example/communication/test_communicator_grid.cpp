#include <iostream>
#include <mpi.h>

#include "communicator_grid.h"

int main() {
  MPI_Init(nullptr, nullptr);

  using namespace dlaf::comm;

  {
    Communicator world;
    CommunicatorGrid grid(world, computeGridDims(world.size()));

    std::cout << "grid" << std::endl;
    std::cout << "(" << grid.rank().row() << "; " << grid.rank().col() << ")" << std::endl;
    std::cout << grid.rows() << "x" << grid.cols() << std::endl;
  }

  MPI_Finalize();

  return 0;
}
