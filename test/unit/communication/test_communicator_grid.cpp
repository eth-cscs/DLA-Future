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

const auto valid_orderings = ::testing::Values(Ordering::RowMajor, Ordering::ColumnMajor);

void test_grid_communication(CommunicatorGrid& grid) {
  if (MPI_COMM_NULL == grid.rowCommunicator() || MPI_COMM_NULL == grid.colCommunicator())
    return;

  const int buffer_send = 1;
  int buffer, buffer_recv;

  // All Communication
  if (grid.rankFullCommunicator(Index2D{0, 0}) == 0)
    buffer = 13;

  MPI_Bcast(&buffer, 1, MPI_INT, 0, grid.fullCommunicator());
  EXPECT_EQ(buffer, 13);

  buffer_recv = 0;
  MPI_Allreduce(&buffer_send, &buffer_recv, 1, MPI_INT, MPI_SUM, grid.fullCommunicator());
  EXPECT_EQ(buffer_recv, grid.size().rows() * grid.size().cols());

  // Row Communication
  if (grid.rank().col() == 0)
    buffer = grid.rank().row();

  MPI_Bcast(&buffer, 1, MPI_INT, 0, grid.rowCommunicator());
  EXPECT_EQ(buffer, grid.rank().row());

  buffer_recv = 0;
  MPI_Allreduce(&buffer_send, &buffer_recv, 1, MPI_INT, MPI_SUM, grid.rowCommunicator());
  EXPECT_EQ(buffer_recv, grid.rowCommunicator().size());

  // Column Communication
  if (grid.rank().row() == 0)
    buffer = grid.rank().col();

  MPI_Bcast(&buffer, 1, MPI_INT, 0, grid.colCommunicator());
  EXPECT_EQ(buffer, grid.rank().col());

  buffer_recv = 0;
  MPI_Allreduce(&buffer_send, &buffer_recv, 1, MPI_INT, MPI_SUM, grid.colCommunicator());
  EXPECT_EQ(buffer_recv, grid.colCommunicator().size());
}

void check_rank_full_communicator(const CommunicatorGrid& grid) {
  // Checks the function rank_full_communicator
  // If the rank is not in the grid:
  //  - all coords must return -1
  // If the rank is in the grid:
  //  - Check that every coords returns a valid distinct rank
  std::set<dlaf::comm::IndexT_MPI> ranks;

  for (int c = 0; c < grid.size().cols(); ++c) {
    for (int r = 0; r < grid.size().rows(); ++r) {
      // keep track of rank indexes for each coordinate of the grid
      auto rank = grid.rankFullCommunicator({r, c});
      ranks.insert(rank);

      // check that it is a valid rank
      EXPECT_GE(rank, 0);
      EXPECT_LT(rank, grid.size().rows() * grid.size().cols());
    }
  }

  // test that each rank has access to all others
  EXPECT_EQ(ranks.size(), grid.size().rows() * grid.size().cols());
}

class CommunicatorGridTest : public ::testing::TestWithParam<Ordering> {};

TEST_P(CommunicatorGridTest, Copy) {
  Communicator world(MPI_COMM_WORLD);

  auto grid_dims = computeGridDims(NUM_MPI_RANKS);
  int nrows = grid_dims[0];
  int ncols = grid_dims[1];

  CommunicatorGrid grid(world, nrows, ncols, GetParam());

  EXPECT_EQ(NUM_MPI_RANKS, grid.size().rows() * grid.size().cols());
  EXPECT_EQ(nrows, grid.size().rows());
  EXPECT_EQ(ncols, grid.size().cols());

  {
    CommunicatorGrid copy = grid;

    EXPECT_EQ(NUM_MPI_RANKS, copy.size().rows() * copy.size().cols());
    EXPECT_EQ(nrows, copy.size().rows());
    EXPECT_EQ(ncols, copy.size().cols());

    int result;
    MPI_Comm_compare(copy.fullCommunicator(), grid.fullCommunicator(), &result);
    EXPECT_EQ(MPI_IDENT, result);
    EXPECT_NE(MPI_COMM_NULL, copy.fullCommunicator());

    MPI_Comm_compare(copy.rowCommunicator(), grid.rowCommunicator(), &result);
    EXPECT_EQ(MPI_IDENT, result);
    EXPECT_NE(MPI_COMM_NULL, copy.rowCommunicator());

    MPI_Comm_compare(copy.colCommunicator(), grid.colCommunicator(), &result);
    EXPECT_EQ(MPI_IDENT, result);
    EXPECT_NE(MPI_COMM_NULL, copy.colCommunicator());

    test_grid_communication(copy);
  }

  EXPECT_EQ(NUM_MPI_RANKS, grid.size().rows() * grid.size().cols());
  EXPECT_EQ(nrows, grid.size().rows());
  EXPECT_EQ(ncols, grid.size().cols());

  EXPECT_NE(MPI_COMM_NULL, grid.fullCommunicator());
  EXPECT_NE(MPI_COMM_NULL, grid.rowCommunicator());
  EXPECT_NE(MPI_COMM_NULL, grid.colCommunicator());

  test_grid_communication(grid);
}

TEST_P(CommunicatorGridTest, ConstructorWithParams) {
  Communicator world(MPI_COMM_WORLD);

  auto grid_dims = computeGridDims(NUM_MPI_RANKS);
  int nrows = grid_dims[0];
  int ncols = grid_dims[1];

  CommunicatorGrid grid(world, nrows, ncols, GetParam());

  EXPECT_EQ(NUM_MPI_RANKS, grid.size().rows() * grid.size().cols());
  EXPECT_EQ(nrows, grid.size().rows());
  EXPECT_EQ(ncols, grid.size().cols());

  test_grid_communication(grid);
}

INSTANTIATE_TEST_SUITE_P(ConstructorWithParams, CommunicatorGridTest, valid_orderings);

TEST_P(CommunicatorGridTest, ConstructorWithArray) {
  Communicator world(MPI_COMM_WORLD);

  const std::array<IndexT_MPI, 2>& grid_dims = computeGridDims(NUM_MPI_RANKS);
  CommunicatorGrid grid(world, grid_dims, GetParam());

  EXPECT_EQ(NUM_MPI_RANKS, grid.size().rows() * grid.size().cols());
  EXPECT_EQ(grid_dims[0], grid.size().rows());
  EXPECT_EQ(grid_dims[1], grid.size().cols());

  test_grid_communication(grid);
}

INSTANTIATE_TEST_SUITE_P(ConstructorWithArray, CommunicatorGridTest, valid_orderings);

TEST_P(CommunicatorGridTest, ConstructorIncomplete) {
  static_assert(NUM_MPI_RANKS > 1, "There must be at least 2 ranks");

  std::array<int, 2> grid_dims{NUM_MPI_RANKS - 1, 1};

  Communicator world(MPI_COMM_WORLD);
  CommunicatorGrid incomplete_grid(world, grid_dims, GetParam());

  if (world.rank() != NUM_MPI_RANKS - 1) {  // ranks in the grid
    auto coords = dlaf::common::computeCoords(GetParam(), world.rank(), Size2D(grid_dims));

    EXPECT_EQ(NUM_MPI_RANKS - 1, incomplete_grid.size().rows());
    EXPECT_EQ(1, incomplete_grid.size().cols());

    EXPECT_NE(MPI_COMM_NULL, incomplete_grid.fullCommunicator());
    EXPECT_NE(MPI_COMM_NULL, incomplete_grid.rowCommunicator());
    EXPECT_NE(MPI_COMM_NULL, incomplete_grid.colCommunicator());

    check_rank_full_communicator(incomplete_grid);
    EXPECT_EQ(coords.row(), incomplete_grid.rank().row());
    EXPECT_EQ(coords.col(), incomplete_grid.rank().col());

    EXPECT_EQ(coords.col(), incomplete_grid.rowCommunicator().rank());
    EXPECT_EQ(coords.row(), incomplete_grid.colCommunicator().rank());
  }
  else {  // last rank is not in the grid
    EXPECT_EQ(0, incomplete_grid.size().rows());
    EXPECT_EQ(0, incomplete_grid.size().cols());

    EXPECT_EQ(MPI_COMM_NULL, incomplete_grid.fullCommunicator());
    EXPECT_EQ(MPI_COMM_NULL, incomplete_grid.rowCommunicator());
    EXPECT_EQ(MPI_COMM_NULL, incomplete_grid.colCommunicator());

    EXPECT_EQ(-1, incomplete_grid.rank().row());
    EXPECT_EQ(-1, incomplete_grid.rank().col());

    EXPECT_EQ(MPI_UNDEFINED, incomplete_grid.fullCommunicator().rank());
    EXPECT_EQ(MPI_UNDEFINED, incomplete_grid.rowCommunicator().rank());
    EXPECT_EQ(MPI_UNDEFINED, incomplete_grid.colCommunicator().rank());
  }

  test_grid_communication(incomplete_grid);
}

INSTANTIATE_TEST_SUITE_P(ConstructorIncomplete, CommunicatorGridTest, valid_orderings);

TEST_P(CommunicatorGridTest, Rank) {
  auto grid_dims = computeGridDims(NUM_MPI_RANKS);

  auto grid_area = 1;
  for (auto dim : grid_dims) {
    ASSERT_NE(1, dim);
    grid_area *= dim;
  }
  ASSERT_EQ(NUM_MPI_RANKS, grid_area);

  Communicator world(MPI_COMM_WORLD);
  CommunicatorGrid complete_grid(world, grid_dims, GetParam());

  EXPECT_EQ(grid_dims[0], complete_grid.size().rows());
  EXPECT_EQ(grid_dims[1], complete_grid.size().cols());

  auto coords = dlaf::common::computeCoords(GetParam(), world.rank(), Size2D(grid_dims));

  check_rank_full_communicator(complete_grid);
  EXPECT_EQ(coords.row(), complete_grid.rank().row());
  EXPECT_EQ(coords.col(), complete_grid.rank().col());

  EXPECT_EQ(coords.col(), complete_grid.rowCommunicator().rank());
  EXPECT_EQ(coords.row(), complete_grid.colCommunicator().rank());
}

INSTANTIATE_TEST_SUITE_P(Rank, CommunicatorGridTest, valid_orderings);
