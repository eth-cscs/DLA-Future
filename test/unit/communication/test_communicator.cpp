//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/communication/communicator.h"

#include <mpi.h>
#include <gtest/gtest.h>

using namespace dlaf::comm;

TEST(Communicator, ConstructorDefault) {
  Communicator world;

  int result;
  MPI_Comm_compare(MPI_COMM_WORLD, static_cast<MPI_Comm>(world), &result);
  EXPECT_EQ(MPI_IDENT, result);

  EXPECT_EQ(world.size(), NUM_MPI_RANKS);
  EXPECT_LT(world.rank(), world.size());
  EXPECT_GE(world.rank(), 0);
}

TEST(Communicator, Constructor) {
  // at least 3 ranks, 2 will be grouped and the rest will be left outside
  ASSERT_GT(NUM_MPI_RANKS, 2);

  Communicator world;

  // split rule (just for 2 ranks will be part of the new communicator)
  int color = MPI_UNDEFINED;
  int key = MPI_UNDEFINED;
  if (world.rank() < 2) {
    color = 0;
    key = !world.rank();    // invert rank
  }

  MPI_Comm new_communicator;
  MPI_CALL(MPI_Comm_split(
    static_cast<MPI_Comm>(Communicator()),
    color,
    key,
    &new_communicator
  ));

  Communicator new_comm(new_communicator);

  // ranks in the new communicator
  if (world.rank() < 2) {
    // check that it is the same communicator as in MPI
    int result;
    MPI_Comm_compare(new_communicator, static_cast<MPI_Comm>(new_comm), &result);
    EXPECT_EQ(MPI_IDENT, result);

    // check that that new communicator size consistency and correctness
    EXPECT_LT(new_comm.size(), NUM_MPI_RANKS);
    EXPECT_EQ(new_comm.size(), 2);

    // check rank consistency
    EXPECT_LT(new_comm.rank(), new_comm.size());
    EXPECT_GE(new_comm.rank(), 0);

    // check rank correctness
    EXPECT_NE(new_comm.rank(), world.rank());
    EXPECT_EQ(new_comm.rank(), !world.rank());
  }
  // ranks oustide the new communicator
  else {
    // check that new communicator is not valid
    EXPECT_EQ(MPI_COMM_NULL, static_cast<MPI_Comm>(new_comm));

    // check rank correctness
    EXPECT_EQ(new_comm.rank(), MPI_UNDEFINED);
    EXPECT_EQ(new_comm.size(), 0);
  }

  // check that in the world nothing is changed
  EXPECT_EQ(world.size(), NUM_MPI_RANKS);
  EXPECT_LT(world.rank(), world.size());
  EXPECT_GE(world.rank(), 0);
}
