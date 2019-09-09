//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/communicator/communicator.h"

#include <mpi.h>
#include <gtest/gtest.h>

TEST(Communicator, basic) {
  dlaf::comm::Communicator world;

  EXPECT_EQ(world.size(), NUM_MPI_RANKS);
}
