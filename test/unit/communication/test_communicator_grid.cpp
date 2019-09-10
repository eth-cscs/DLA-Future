//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/communication/communicator_grid.h"

#include <mpi.h>
#include <gtest/gtest.h>

TEST(CommunicatorGrid, basic) {
  using namespace dlaf::comm;

  Communicator world;
  CommunicatorGrid grid(world, computeGridDims(world.size()));

  EXPECT_EQ(grid.rows() * grid.cols(), NUM_MPI_RANKS);

  // std::cout << "(" << grid.rank().row() << "; " << grid.rank().col() << ")" << std::endl;
}
