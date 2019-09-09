#include "dlaf/communicator/communicator.h"

#include <mpi.h>
#include <gtest/gtest.h>

TEST(Communicator, basic) {
  dlaf::comm::Communicator world;

  EXPECT_EQ(world.size(), NUM_MPI_RANKS);
}
