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

#include <gtest/gtest.h>
#include <mpi.h>

using namespace dlaf::comm;

class CommunicatorTest : public ::testing::Test {
protected:
  ~CommunicatorTest() noexcept {};

  void SetUp() noexcept(false) override {
    world = Communicator(MPI_COMM_WORLD);

    if (rankInGroup()) {
      color = 0;
      key = world.rank() / 2;
    }

    MPI_Comm odd_mpi_comm;
    MPI_CALL(MPI_Comm_split(world, color, key, &odd_mpi_comm));

    if (rankInGroup()) {
      ASSERT_NE(MPI_COMM_NULL, odd_mpi_comm);
      odd_comm = createCommunicator(odd_mpi_comm);
    }
    else {
      ASSERT_EQ(MPI_COMM_NULL, odd_mpi_comm);
      ASSERT_EQ(MPI_COMM_NULL, odd_comm);
    }
  }

  bool rankInGroup() const {
    return world.rank() % 2 == 1;
  }

  Communicator world;
  Communicator odd_comm;
  int color = MPI_UNDEFINED;
  int key = MPI_UNDEFINED;

private:
  virtual Communicator createCommunicator(MPI_Comm mpi_communicator) = 0;
};

class CommunicatorTestNotManaged : public CommunicatorTest {
protected:
  virtual Communicator createCommunicator(MPI_Comm mpi_communicator) override {
    return Communicator(mpi_communicator);
  }

  void TearDown() noexcept(false) override {
    if (MPI_COMM_NULL != odd_comm)
      MPI_CALL(MPI_Comm_free(&odd_comm));
  }
};

class CommunicatorTestManaged : public CommunicatorTest {
protected:
  virtual Communicator createCommunicator(MPI_Comm mpi_communicator) override {
    return Communicator(mpi_communicator, Managed());
  }
};

TEST(Communicator, ConstructorDefault) {
  Communicator comm_null;

  EXPECT_EQ(MPI_COMM_NULL, comm_null);

  EXPECT_EQ(comm_null.size(), 0);
  EXPECT_EQ(comm_null.rank(), MPI_UNDEFINED);
}

TEST_F(CommunicatorTestNotManaged, Rank) {
  if (rankInGroup()) {
    // check new communicator size consistency and correctness
    EXPECT_LT(odd_comm.size(), NUM_MPI_RANKS);
    EXPECT_EQ(odd_comm.size(), (NUM_MPI_RANKS + 1) / 2);

    // check rank consistency
    EXPECT_LT(odd_comm.rank(), odd_comm.size());
    EXPECT_GE(odd_comm.rank(), 0);

    // check rank correctness
    EXPECT_NE(odd_comm.rank(), world.rank());
    EXPECT_EQ(odd_comm.rank(), world.rank() / 2);
  }
  else {
    // check that new communicator is not valid
    EXPECT_EQ(MPI_COMM_NULL, odd_comm);

    // check rank correctness
    EXPECT_EQ(odd_comm.rank(), MPI_UNDEFINED);
    EXPECT_EQ(odd_comm.size(), 0);
  }

  // check that in the world nothing is changed
  EXPECT_EQ(world.size(), NUM_MPI_RANKS);
  EXPECT_LT(world.rank(), world.size());
  EXPECT_GE(world.rank(), 0);
}

TEST_F(CommunicatorTestManaged, Rank) {
  if (rankInGroup()) {
    // check new communicator size consistency and correctness
    EXPECT_LT(odd_comm.size(), NUM_MPI_RANKS);
    EXPECT_EQ(odd_comm.size(), (NUM_MPI_RANKS + 1) / 2);

    // check rank consistency
    EXPECT_LT(odd_comm.rank(), odd_comm.size());
    EXPECT_GE(odd_comm.rank(), 0);

    // check rank correctness
    EXPECT_NE(odd_comm.rank(), world.rank());
    EXPECT_EQ(odd_comm.rank(), world.rank() / 2);
  }
  else {
    // check that new communicator is not valid
    EXPECT_EQ(MPI_COMM_NULL, odd_comm);

    // check rank correctness
    EXPECT_EQ(odd_comm.rank(), MPI_UNDEFINED);
    EXPECT_EQ(odd_comm.size(), 0);
  }

  // check that in the world nothing is changed
  EXPECT_EQ(world.size(), NUM_MPI_RANKS);
  EXPECT_LT(world.rank(), world.size());
  EXPECT_GE(world.rank(), 0);
}

TEST_F(CommunicatorTestNotManaged, Copy) {
  Communicator copy = odd_comm;

  if (rankInGroup()) {
    int result;
    MPI_Comm_compare(copy, odd_comm, &result);
    EXPECT_EQ(MPI_IDENT, result);

    // check new communicator size consistency and correctness
    EXPECT_EQ(odd_comm.size(), copy.size());

    // check rank consistency
    EXPECT_LT(copy.rank(), copy.size());
    EXPECT_GE(copy.rank(), 0);

    // check rank correctness
    EXPECT_EQ(odd_comm.rank(), copy.rank());
  }
  else {
    // check that new communicator is not valid
    EXPECT_EQ(MPI_COMM_NULL, odd_comm);

    // check rank correctness
    EXPECT_EQ(copy.rank(), MPI_UNDEFINED);
    EXPECT_EQ(copy.size(), 0);
  }

  // check that in the world nothing is changed
  EXPECT_EQ(world.size(), NUM_MPI_RANKS);
  EXPECT_LT(world.rank(), world.size());
  EXPECT_GE(world.rank(), 0);
}

TEST_F(CommunicatorTestManaged, Copy) {
  if (rankInGroup()) {
    // create a copy (and check it is an handler for the same)
    MPI_Comm original_comm = odd_comm;

    {
      Communicator copy = odd_comm;

      int result;
      MPI_Comm_compare(copy, odd_comm, &result);
      EXPECT_EQ(MPI_IDENT, result);

      // check new communicator size consistency and correctness
      EXPECT_EQ(odd_comm.size(), copy.size());

      // check rank consistency
      EXPECT_LT(copy.rank(), copy.size());
      EXPECT_GE(copy.rank(), 0);

      // check rank correctness
      EXPECT_EQ(odd_comm.rank(), copy.rank());
    }

    // TODO check that copy have been released

    // once the duplicated is destroyed (out-of-scope), check the previous is still valid
    int result;
    MPI_CALL(MPI_Comm_compare(odd_comm, original_comm, &result));
    EXPECT_EQ(MPI_IDENT, result);

    // check new communicator size consistency and correctness
    EXPECT_LT(odd_comm.size(), NUM_MPI_RANKS);
    EXPECT_EQ(odd_comm.size(), (NUM_MPI_RANKS + 1) / 2);

    // check rank consistency
    EXPECT_LT(odd_comm.rank(), odd_comm.size());
    EXPECT_GE(odd_comm.rank(), 0);

    // check rank correctness
    EXPECT_NE(odd_comm.rank(), world.rank());
    EXPECT_EQ(odd_comm.rank(), world.rank() / 2);
  }
  else {
    // check that new communicator is not valid
    EXPECT_EQ(MPI_COMM_NULL, odd_comm);

    // check rank correctness
    EXPECT_EQ(odd_comm.rank(), MPI_UNDEFINED);
    EXPECT_EQ(odd_comm.size(), 0);
  }

  // check that in the world nothing is changed
  EXPECT_EQ(world.size(), NUM_MPI_RANKS);
  EXPECT_LT(world.rank(), world.size());
  EXPECT_GE(world.rank(), 0);
}
