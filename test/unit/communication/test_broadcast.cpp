#include <gtest/gtest.h>

#include "dlaf/communication/functions.h"

using dlaf::comm::Communicator;

class BroadcastTest : public ::testing::Test {
protected:
  void SetUp() override {
    world = Communicator(MPI_COMM_WORLD);

    color = world.rank() % 2;
    key = world.rank() / 2;

    MPI_Comm mpi_splitted_comm;
    MPI_Comm_split(world, color, key, &mpi_splitted_comm);

    ASSERT_NE(MPI_COMM_NULL, mpi_splitted_comm);
    splitted_comm = Communicator(mpi_splitted_comm);
  }

  void TearDown() override {
    if (MPI_COMM_NULL != splitted_comm)
      MPI_Comm_free(&splitted_comm);
  }

  bool isMasterInSplitted() {
    return world.rank() == color;
  }

  Communicator world;
  Communicator splitted_comm;

  int color = MPI_UNDEFINED;
  int key = MPI_UNDEFINED;
};

TEST_F(BroadcastTest, Broadcast_ClassicAPI) {
  auto broadcaster = 0;
  auto where = splitted_comm;

  int what;
  if (isMasterInSplitted())
    what = color;
  else
    what = -1;

  dlaf::comm::bcast(broadcaster, what, where);
  EXPECT_EQ(color, what);
}

TEST_F(BroadcastTest, Broadcast_ClassicAPI_Splitted) {
  auto broadcaster = 0;
  auto where = splitted_comm;

  if (isMasterInSplitted())
    dlaf::comm::bcast(broadcaster, color, where);
  else {
    int what;
    dlaf::comm::bcast(broadcaster, what, where);
    EXPECT_EQ(color, what);
  }
}

TEST_F(BroadcastTest, Broadcast_NewAPI) {
  auto broadcaster = 0;
  auto where = splitted_comm;

  if (isMasterInSplitted())
    dlaf::comm::broadcast::send(color, where);
  else {
    int what;
    dlaf::comm::broadcast::receive_from(broadcaster, what, where);
    EXPECT_EQ(color, what);
  }
}

TEST_F(BroadcastTest, AsyncBroadcast_ClassicAPI) {
  auto broadcaster = 0;
  auto where = splitted_comm;

  bool waited;
  auto what_to_do_before_retesting = [&waited]() { waited = true; };

  int what;
  if (isMasterInSplitted())
    what = color;
  else
    what = -1;

  dlaf::comm::async_bcast(broadcaster, what, where, what_to_do_before_retesting);
  EXPECT_EQ(color, what);
}

TEST_F(BroadcastTest, AsyncBroadcast_ClassicAPI_Splitted) {
  auto broadcaster = 0;
  auto where = splitted_comm;

  bool waited;
  auto what_to_do_before_retesting = [&waited]() { waited = true; };

  if (isMasterInSplitted())
    dlaf::comm::async_bcast(broadcaster, color, where, what_to_do_before_retesting);
  else {
    int what;
    dlaf::comm::async_bcast(broadcaster, what, where, what_to_do_before_retesting);
    EXPECT_EQ(color, what);
  }
}

TEST_F(BroadcastTest, AsyncBroadcast_NewAPI) {
  auto broadcaster = 0;
  auto where = splitted_comm;

  bool waited;
  auto what_to_do_before_retesting = [&waited]() { waited = true; };

  if (isMasterInSplitted())
    dlaf::comm::async_broadcast::send(color, where, what_to_do_before_retesting);
  else {
    int what;
    dlaf::comm::async_broadcast::receive_from(broadcaster, what, where, what_to_do_before_retesting);
    EXPECT_EQ(color, what);
  }
}
