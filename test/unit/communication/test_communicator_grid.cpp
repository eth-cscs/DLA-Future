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

#include <gtest/gtest.h>
#include <mpi.h>

using dlaf::common::LeadingDimension;
using namespace dlaf::comm;

struct coords_t {
  const int row;
  const int col;
};

class CommunicatorGridTest : public ::testing::TestWithParam<LeadingDimension> {};

TEST_P(CommunicatorGridTest, ConstructorWithParams) {
  Communicator world(MPI_COMM_WORLD);

  auto grid_dims = computeGridDims(NUM_MPI_RANKS);
  int nrows = grid_dims[0];
  int ncols = grid_dims[1];

  CommunicatorGrid grid(world, nrows, ncols, GetParam());

  EXPECT_EQ(grid.rows() * grid.cols(), NUM_MPI_RANKS);
  EXPECT_EQ(grid.rows(), nrows);
  EXPECT_EQ(grid.cols(), ncols);
}

INSTANTIATE_TEST_CASE_P(ConstructorWithParams, CommunicatorGridTest,
                        ::testing::Values(LeadingDimension::Row, LeadingDimension::Column));

TEST_P(CommunicatorGridTest, ConstructorWithArray) {
  Communicator world(MPI_COMM_WORLD);

  auto grid_dims = computeGridDims(NUM_MPI_RANKS);
  CommunicatorGrid grid(world, grid_dims, GetParam());

  EXPECT_EQ(grid.rows() * grid.cols(), NUM_MPI_RANKS);
  EXPECT_EQ(grid.rows(), grid_dims[0]);
  EXPECT_EQ(grid.cols(), grid_dims[1]);
}

INSTANTIATE_TEST_CASE_P(ConstructorWithArray, CommunicatorGridTest,
                        ::testing::Values(LeadingDimension::Row, LeadingDimension::Column));

TEST_P(CommunicatorGridTest, ConstructorOverfit) {
  Communicator world(MPI_COMM_WORLD);

  EXPECT_ANY_THROW(CommunicatorGrid grid(world, NUM_MPI_RANKS, 2, GetParam()));
}

INSTANTIATE_TEST_CASE_P(ConstructorOverfit, CommunicatorGridTest,
                        ::testing::Values(LeadingDimension::Row, LeadingDimension::Column));

TEST_P(CommunicatorGridTest, ConstructorIncomplete) {
  static_assert(NUM_MPI_RANKS > 1, "There must be at least 2 ranks");

  std::array<int, 2> grid_dims{NUM_MPI_RANKS - 1, 1};

  Communicator world(MPI_COMM_WORLD);
  CommunicatorGrid incomplete_grid(world, grid_dims, GetParam());

  if (world.rank() != NUM_MPI_RANKS - 1) {  // ranks in the grid
    EXPECT_EQ(incomplete_grid.rows(), NUM_MPI_RANKS - 1);
    EXPECT_EQ(incomplete_grid.cols(), 1);

    EXPECT_NE(incomplete_grid.row(), MPI_COMM_NULL);
    EXPECT_NE(incomplete_grid.col(), MPI_COMM_NULL);

    auto coords = dlaf::common::computeCoords(GetParam(), world.rank(), grid_dims);

    EXPECT_EQ(coords.row(), incomplete_grid.rank().row());
    EXPECT_EQ(coords.col(), incomplete_grid.rank().col());

    EXPECT_EQ(coords.col(), incomplete_grid.row().rank());
    EXPECT_EQ(coords.row(), incomplete_grid.col().rank());
  }
  else {  // last rank is not in the grid
    EXPECT_EQ(incomplete_grid.rows(), 0);
    EXPECT_EQ(incomplete_grid.cols(), 0);

    EXPECT_EQ(incomplete_grid.row(), MPI_COMM_NULL);
    EXPECT_EQ(incomplete_grid.col(), MPI_COMM_NULL);

    EXPECT_EQ(incomplete_grid.rank().row(), -1);
    EXPECT_EQ(incomplete_grid.rank().col(), -1);

    EXPECT_EQ(incomplete_grid.row().rank(), MPI_UNDEFINED);
    EXPECT_EQ(incomplete_grid.col().rank(), MPI_UNDEFINED);
  }
}

INSTANTIATE_TEST_CASE_P(ConstructorIncomplete, CommunicatorGridTest,
                        ::testing::Values(LeadingDimension::Row, LeadingDimension::Column));

TEST_P(CommunicatorGridTest, Rank) {
  auto grid_dims = computeGridDims(NUM_MPI_RANKS);

  auto grid_area = 1;
  for (auto dim : grid_dims) {
    ASSERT_NE(dim, 1);
    grid_area *= dim;
  }
  ASSERT_EQ(grid_area, NUM_MPI_RANKS);

  Communicator world(MPI_COMM_WORLD);
  CommunicatorGrid complete_grid(world, grid_dims, GetParam());

  EXPECT_EQ(complete_grid.rows(), grid_dims[0]);
  EXPECT_EQ(complete_grid.cols(), grid_dims[1]);

  auto coords = computeCoords(GetParam(), world.rank(), grid_dims);

  EXPECT_EQ(coords.row(), complete_grid.rank().row());
  EXPECT_EQ(coords.col(), complete_grid.rank().col());

  EXPECT_EQ(coords.col(), complete_grid.row().rank());
  EXPECT_EQ(coords.row(), complete_grid.col().rank());
}

INSTANTIATE_TEST_CASE_P(Rank, CommunicatorGridTest,
                        ::testing::Values(LeadingDimension::Row, LeadingDimension::Column));
