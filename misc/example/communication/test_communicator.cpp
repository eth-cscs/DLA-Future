#include <iostream>
#include <mpi.h>

#include "communicator.h"

int main() {
  MPI_Init(nullptr, nullptr);

  dlaf::comm::Communicator world;

  std::cout << world.rank() << "/" << world.size() << std::endl;

  MPI_Finalize();
  return 0;
}
