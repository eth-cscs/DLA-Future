#include "dlaf/communicator/communicator_grid.h"

#include <mpi.h>
#include <gtest/gtest.h>

TEST(CommunicatorGrid, basic) {
  using namespace dlaf::comm;

  Communicator world;
  CommunicatorGrid grid(world, computeGridDims(world.size()));

  EXPECT_EQ(grid.rows() * grid.cols(), NUM_MPI_RANKS);

  // std::cout << "(" << grid.rank().row() << "; " << grid.rank().col() << ")" << std::endl;
}
