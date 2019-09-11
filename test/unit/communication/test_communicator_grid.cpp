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

using namespace dlaf::comm;

TEST(CommunicatorGrid, ConstructorWithParams) {
  Communicator world(MPI_COMM_WORLD);

  auto grid_dims = computeGridDims(NUM_MPI_RANKS);
  int nrows = grid_dims[0];
  int ncols = grid_dims[1];

  CommunicatorGrid grid(world, nrows, ncols);

  EXPECT_EQ(grid.rows() * grid.cols(), NUM_MPI_RANKS);
  EXPECT_EQ(grid.rows(), nrows);
  EXPECT_EQ(grid.cols(), ncols);
}

TEST(CommunicatorGrid, ConstructorWithArray) {
  Communicator world(MPI_COMM_WORLD);

  auto grid_dims = computeGridDims(NUM_MPI_RANKS);
  CommunicatorGrid grid(world, grid_dims);

  EXPECT_EQ(grid.rows() * grid.cols(), NUM_MPI_RANKS);
  EXPECT_EQ(grid.rows(), grid_dims[0]);
  EXPECT_EQ(grid.cols(), grid_dims[1]);
}

TEST(CommunicatorGrid, ConstructorIncomplete) {
  static_assert(NUM_MPI_RANKS > 1, "There must be at least 2 ranks");

  Communicator world(MPI_COMM_WORLD);
  CommunicatorGrid incomplete_grid(world, NUM_MPI_RANKS - 1, 1);

  if (world.rank() != NUM_MPI_RANKS - 1) { // ranks in the grid
    EXPECT_EQ(incomplete_grid.rows(), NUM_MPI_RANKS - 1);
    EXPECT_EQ(incomplete_grid.cols(), 1);

    EXPECT_NE(incomplete_grid.all(), MPI_COMM_NULL);
    EXPECT_NE(incomplete_grid.row(), MPI_COMM_NULL);
    EXPECT_NE(incomplete_grid.col(), MPI_COMM_NULL);

    auto rank_row = world.rank() / incomplete_grid.cols();
    auto rank_col = world.rank() % incomplete_grid.cols();

    EXPECT_EQ(rank_row, incomplete_grid.rank().row());
    EXPECT_EQ(rank_col, incomplete_grid.rank().col());

    EXPECT_EQ(rank_row, incomplete_grid.row().rank());
    EXPECT_EQ(rank_col, incomplete_grid.col().rank());
  }
  else {  // last rank is not in the grid
    EXPECT_EQ(incomplete_grid.rows(), MPI_UNDEFINED);
    EXPECT_EQ(incomplete_grid.cols(), MPI_UNDEFINED);

    EXPECT_EQ(incomplete_grid.all(), MPI_COMM_NULL);
    EXPECT_EQ(incomplete_grid.row(), MPI_COMM_NULL);
    EXPECT_EQ(incomplete_grid.col(), MPI_COMM_NULL);

    EXPECT_EQ(incomplete_grid.rank().row(), -1);
    EXPECT_EQ(incomplete_grid.rank().col(), -1);

    EXPECT_EQ(incomplete_grid.row().rank(), MPI_UNDEFINED);
    EXPECT_EQ(incomplete_grid.col().rank(), MPI_UNDEFINED);
  }
}

TEST(CommunicatorGrid, Rank) {
  auto grid_dims = computeGridDims(NUM_MPI_RANKS);

  auto grid_area = 1;
  for(auto dim : grid_dims) {
    ASSERT_NE(dim, 1);
    grid_area *= dim;
  }
  ASSERT_EQ(grid_area, NUM_MPI_RANKS);

  Communicator world(MPI_COMM_WORLD);
  CommunicatorGrid complete_grid(world, grid_dims);

  EXPECT_EQ(complete_grid.rows(), grid_dims[0]);
  EXPECT_EQ(complete_grid.cols(), grid_dims[1]);

  auto rank_row = world.rank() / complete_grid.cols();
  auto rank_col = world.rank() % complete_grid.cols();

  EXPECT_EQ(rank_row, complete_grid.rank().row());
  EXPECT_EQ(rank_col, complete_grid.rank().col());

  // TODO equivalent call...redundancy?!
  EXPECT_EQ(rank_row, complete_grid.row().rank());
  EXPECT_EQ(rank_col, complete_grid.col().rank());
}
