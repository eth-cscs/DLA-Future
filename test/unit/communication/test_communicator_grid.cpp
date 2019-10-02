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
  if (MPI_COMM_NULL == grid.row() || MPI_COMM_NULL == grid.col())
    return;

  int buffer;
  int buffer_send = 1, buffer_recv;

  // Row Communication
  if (grid.rank().col() == 0)
    buffer = grid.rank().row();

  MPI_Bcast(&buffer, 1, MPI_INT, 0, grid.row());
  EXPECT_EQ(buffer, grid.rank().row());

  buffer_recv = 0;
  MPI_Allreduce(&buffer_send, &buffer_recv, 1, MPI_INT, MPI_SUM, grid.row());
  EXPECT_EQ(buffer_recv, grid.row().size());

  // Column Communication
  if (grid.rank().row() == 0)
    buffer = grid.rank().col();

  MPI_Bcast(&buffer, 1, MPI_INT, 0, grid.col());
  EXPECT_EQ(buffer, grid.rank().col());

  buffer_recv = 0;
  MPI_Allreduce(&buffer_send, &buffer_recv, 1, MPI_INT, MPI_SUM, grid.col());
  EXPECT_EQ(buffer_recv, grid.col().size());
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
    MPI_Comm_compare(copy.row(), grid.row(), &result);
    EXPECT_EQ(MPI_IDENT, result);
    EXPECT_NE(MPI_COMM_NULL, copy.row());

    MPI_Comm_compare(copy.col(), grid.col(), &result);
    EXPECT_EQ(MPI_IDENT, result);
    EXPECT_NE(MPI_COMM_NULL, copy.col());

    test_grid_communication(copy);
  }

  EXPECT_EQ(NUM_MPI_RANKS, grid.size().rows() * grid.size().cols());
  EXPECT_EQ(nrows, grid.size().rows());
  EXPECT_EQ(ncols, grid.size().cols());

  EXPECT_NE(MPI_COMM_NULL, grid.row());
  EXPECT_NE(MPI_COMM_NULL, grid.col());

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

INSTANTIATE_TEST_CASE_P(ConstructorWithParams, CommunicatorGridTest, valid_orderings);

TEST_P(CommunicatorGridTest, ConstructorWithArray) {
  Communicator world(MPI_COMM_WORLD);

  auto grid_dims = computeGridDims(NUM_MPI_RANKS);
  CommunicatorGrid grid(world, grid_dims, GetParam());

  EXPECT_EQ(NUM_MPI_RANKS, grid.size().rows() * grid.size().cols());
  EXPECT_EQ(grid_dims[0], grid.size().rows());
  EXPECT_EQ(grid_dims[1], grid.size().cols());

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
    EXPECT_EQ(NUM_MPI_RANKS - 1, incomplete_grid.size().rows());
    EXPECT_EQ(1, incomplete_grid.size().cols());

    EXPECT_NE(MPI_COMM_NULL, incomplete_grid.row());
    EXPECT_NE(MPI_COMM_NULL, incomplete_grid.col());

    auto coords =
        dlaf::common::computeCoords<CommunicatorGrid::Index2D>(GetParam(), world.rank(), grid_dims);

    EXPECT_EQ(coords.row(), incomplete_grid.rank().row());
    EXPECT_EQ(coords.col(), incomplete_grid.rank().col());

    EXPECT_EQ(coords.col(), incomplete_grid.row().rank());
    EXPECT_EQ(coords.row(), incomplete_grid.col().rank());
  }
  else {  // last rank is not in the grid
    EXPECT_EQ(0, incomplete_grid.size().rows());
    EXPECT_EQ(0, incomplete_grid.size().cols());

    EXPECT_EQ(MPI_COMM_NULL, incomplete_grid.row());
    EXPECT_EQ(MPI_COMM_NULL, incomplete_grid.col());

    EXPECT_EQ(-1, incomplete_grid.rank().row());
    EXPECT_EQ(-1, incomplete_grid.rank().col());

    EXPECT_EQ(MPI_UNDEFINED, incomplete_grid.row().rank());
    EXPECT_EQ(MPI_UNDEFINED, incomplete_grid.col().rank());
  }

  test_grid_communication(incomplete_grid);
}

INSTANTIATE_TEST_CASE_P(ConstructorIncomplete, CommunicatorGridTest, valid_orderings);

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

  auto coords =
      dlaf::common::computeCoords<CommunicatorGrid::Index2D>(GetParam(), world.rank(), grid_dims);

  EXPECT_EQ(coords.row(), complete_grid.rank().row());
  EXPECT_EQ(coords.col(), complete_grid.rank().col());

  EXPECT_EQ(coords.col(), complete_grid.row().rank());
  EXPECT_EQ(coords.row(), complete_grid.col().rank());
}

INSTANTIATE_TEST_CASE_P(Rank, CommunicatorGridTest, valid_orderings);
