//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <unordered_map>

#include <dlaf/common/index2d.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf_c/grid.h>
#include <dlaf_c_test/blacs.h>

#include <gtest/gtest.h>

std::unordered_map<char, dlaf::common::Ordering> ordering = {{'R', dlaf::common::Ordering::RowMajor},
                                                             {'C', dlaf::common::Ordering::ColumnMajor}};

#ifdef DLAF_WITH_SCALAPACK
TEST(GridTest, GridScaLAPACKOrdering) {
  for (const auto& [key, value] : ordering) {
    char order = key;
    EXPECT_EQ(order, key);

    int context;
    Cblacs_get(0, 0, &context);
    Cblacs_gridinit(&context, &order, 2, 3);

    int nprow, npcol, mynprow, mynpcol;
    Cblacs_gridinfo(context, &nprow, &npcol, &mynprow, &mynpcol);

    char rm = grid_ordering(MPI_COMM_WORLD, nprow, npcol, mynprow, mynpcol);
    EXPECT_EQ(rm, key);

    Cblacs_gridexit(context);
  }
}
#endif

TEST(GridTest, GridDLAFOrdering) {
  // TODO
  const std::size_t ncommunicator_pipelines = 5;

  for (const auto [key, value] : ordering) {
    dlaf::comm::Communicator world(MPI_COMM_WORLD);

    dlaf::comm::CommunicatorGrid row_major(world, 2, 3, value, ncommunicator_pipelines);

    char rm = grid_ordering(world, row_major.size().rows(), row_major.size().cols(),
                            row_major.rank().row(), row_major.rank().col());
    EXPECT_EQ(rm, key);
  }
}
