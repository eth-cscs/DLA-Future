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

#include "dlaf_test/util_mpi.h"

using dlaf_test::comm::computeGridDims;
using dlaf::common::Ordering;
using namespace dlaf::comm;

auto valid_orderings = ::testing::Values(Ordering::RowMajor, Ordering::ColumnMajor);

void test_grid_communication(CommunicatorGrid& grid) {
  if (mpi::NULL_COMMUNICATOR == grid.row() || mpi::NULL_COMMUNICATOR == grid.col())
    return;

  int buffer;
  int buffer_send = 1, buffer_recv;

  // Row Communication
  if (grid.rank().col() == 0)
    buffer = grid.rank().row();

  MPI_CALL(MPI_Bcast(&buffer, 1, MPI_INT, 0, grid.row()));
  EXPECT_EQ(buffer, grid.rank().row());

  buffer_recv = 0;
  MPI_CALL(MPI_Allreduce(&buffer_send, &buffer_recv, 1, MPI_INT, MPI_SUM, grid.row()));
  EXPECT_EQ(buffer_recv, grid.row().size());

  // Column Communication
  if (grid.rank().row() == 0)
    buffer = grid.rank().col();

  MPI_CALL(MPI_Bcast(&buffer, 1, MPI_INT, 0, grid.col()));
  EXPECT_EQ(buffer, grid.rank().col());

  buffer_recv = 0;
  MPI_CALL(MPI_Allreduce(&buffer_send, &buffer_recv, 1, MPI_INT, MPI_SUM, grid.col()));
  EXPECT_EQ(buffer_recv, grid.col().size());
}

class CommunicatorGridTest : public ::testing::TestWithParam<Ordering> {};

TEST_P(CommunicatorGridTest, Copy) {
  Communicator world(MPI_COMM_WORLD);

  auto grid_dims = computeGridDims(NUM_MPI_RANKS);
  int nrows = grid_dims[0];
  int ncols = grid_dims[1];

  CommunicatorGrid grid(world, nrows, ncols, GetParam());

  EXPECT_EQ(grid.size().row() * grid.size().col(), NUM_MPI_RANKS);
  EXPECT_EQ(grid.size().row(), nrows);
  EXPECT_EQ(grid.size().col(), ncols);

  {
    CommunicatorGrid copy = grid;

    EXPECT_EQ(copy.size().row() * copy.size().col(), NUM_MPI_RANKS);
    EXPECT_EQ(copy.size().row(), nrows);
    EXPECT_EQ(copy.size().col(), ncols);

    int result;
    MPI_CALL(MPI_Comm_compare(copy.row(), grid.row(), &result));
    EXPECT_EQ(MPI_IDENT, result);
    EXPECT_NE(mpi::NULL_COMMUNICATOR, copy.row());

    MPI_CALL(MPI_Comm_compare(copy.col(), grid.col(), &result));
    EXPECT_EQ(MPI_IDENT, result);
    EXPECT_NE(mpi::NULL_COMMUNICATOR, copy.col());

    test_grid_communication(copy);
  }

  EXPECT_EQ(grid.size().row() * grid.size().col(), NUM_MPI_RANKS);
  EXPECT_EQ(grid.size().row(), nrows);
  EXPECT_EQ(grid.size().col(), ncols);

  EXPECT_NE(mpi::NULL_COMMUNICATOR, grid.row());
  EXPECT_NE(mpi::NULL_COMMUNICATOR, grid.col());

  test_grid_communication(grid);
}

TEST_P(CommunicatorGridTest, ConstructorWithParams) {
  Communicator world(MPI_COMM_WORLD);

  auto grid_dims = computeGridDims(NUM_MPI_RANKS);
  int nrows = grid_dims[0];
  int ncols = grid_dims[1];

  CommunicatorGrid grid(world, nrows, ncols, GetParam());

  EXPECT_EQ(grid.size().row() * grid.size().col(), NUM_MPI_RANKS);
  EXPECT_EQ(grid.size().row(), nrows);
  EXPECT_EQ(grid.size().col(), ncols);

  test_grid_communication(grid);
}

INSTANTIATE_TEST_CASE_P(ConstructorWithParams, CommunicatorGridTest, valid_orderings);

TEST_P(CommunicatorGridTest, ConstructorWithArray) {
  Communicator world(MPI_COMM_WORLD);

  auto grid_dims = computeGridDims(NUM_MPI_RANKS);
  CommunicatorGrid grid(world, grid_dims, GetParam());

  EXPECT_EQ(grid.size().row() * grid.size().col(), NUM_MPI_RANKS);
  EXPECT_EQ(grid.size().row(), grid_dims[0]);
  EXPECT_EQ(grid.size().col(), grid_dims[1]);

  test_grid_communication(grid);
}

INSTANTIATE_TEST_CASE_P(ConstructorWithArray, CommunicatorGridTest, valid_orderings);

TEST_P(CommunicatorGridTest, ConstructorOverfit) {
  Communicator world(MPI_COMM_WORLD);

  EXPECT_THROW(CommunicatorGrid grid(world, NUM_MPI_RANKS, 2, GetParam()), std::invalid_argument);
  EXPECT_THROW(CommunicatorGrid grid(world, NUM_MPI_RANKS + 1, 1, GetParam()), std::invalid_argument);
}

INSTANTIATE_TEST_CASE_P(ConstructorOverfit, CommunicatorGridTest, valid_orderings);

TEST_P(CommunicatorGridTest, ConstructorIncomplete) {
  static_assert(NUM_MPI_RANKS > 1, "There must be at least 2 ranks");

  std::array<int, 2> grid_dims{NUM_MPI_RANKS - 1, 1};

  Communicator world(MPI_COMM_WORLD);
  CommunicatorGrid incomplete_grid(world, grid_dims, GetParam());

  if (world.rank() != NUM_MPI_RANKS - 1) {  // ranks in the grid
    EXPECT_EQ(incomplete_grid.size().row(), NUM_MPI_RANKS - 1);
    EXPECT_EQ(incomplete_grid.size().col(), 1);

    EXPECT_NE(incomplete_grid.row(), MPI_COMM_NULL);
    EXPECT_NE(incomplete_grid.col(), MPI_COMM_NULL);

    auto coords = dlaf::common::computeCoords(GetParam(), world.rank(), grid_dims);

    EXPECT_EQ(coords.row(), incomplete_grid.rank().row());
    EXPECT_EQ(coords.col(), incomplete_grid.rank().col());

    EXPECT_EQ(coords.col(), incomplete_grid.row().rank());
    EXPECT_EQ(coords.row(), incomplete_grid.col().rank());
  }
  else {  // last rank is not in the grid
    EXPECT_EQ(incomplete_grid.size().row(), 0);
    EXPECT_EQ(incomplete_grid.size().col(), 0);

    EXPECT_EQ(incomplete_grid.row(), MPI_COMM_NULL);
    EXPECT_EQ(incomplete_grid.col(), MPI_COMM_NULL);

    EXPECT_EQ(incomplete_grid.rank().row(), -1);
    EXPECT_EQ(incomplete_grid.rank().col(), -1);

    EXPECT_EQ(incomplete_grid.row().rank(), MPI_UNDEFINED);
    EXPECT_EQ(incomplete_grid.col().rank(), MPI_UNDEFINED);
  }

  test_grid_communication(incomplete_grid);
}

INSTANTIATE_TEST_CASE_P(ConstructorIncomplete, CommunicatorGridTest, valid_orderings);

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

  EXPECT_EQ(complete_grid.size().row(), grid_dims[0]);
  EXPECT_EQ(complete_grid.size().col(), grid_dims[1]);

  auto coords = computeCoords(GetParam(), world.rank(), grid_dims);

  EXPECT_EQ(coords.row(), complete_grid.rank().row());
  EXPECT_EQ(coords.col(), complete_grid.rank().col());

  EXPECT_EQ(coords.col(), complete_grid.row().rank());
  EXPECT_EQ(coords.row(), complete_grid.col().rank());
}

INSTANTIATE_TEST_CASE_P(Rank, CommunicatorGridTest, valid_orderings);
