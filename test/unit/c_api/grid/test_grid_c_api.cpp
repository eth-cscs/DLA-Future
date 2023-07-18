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
#include <dlaf_c/grid.h>

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>

std::unordered_map<char, dlaf::common::Ordering> ordering = {{'R', dlaf::common::Ordering::RowMajor},
                                                             {'C', dlaf::common::Ordering::ColumnMajor}};

#ifdef DLAF_WITH_SCALAPACK
TEST(GridTest, GridScaLAPACKOrdering) {
  for (const auto& [key, value] : ordering) {
    char order = key;

    int context;
    Cblacs_get(0, 0, &context);
    Cblacs_gridinit(&context, &order, 2, 3);

    int nprow, npcol, mynprow, mynpcol;
    Cblacs_gridinfo(context, &nprow, &npcol, &mynprow, &mynpcol);

    int system_context;
    int get_blacs_contxt = 10;  // SGET_BLACSCONTXT == 10
    Cblacs_get(context, get_blacs_contxt, &system_context);

    MPI_Comm comm = Cblacs2sys_handle(system_context);

    char rm = grid_ordering(comm, nprow, npcol, mynprow, mynpcol);
    EXPECT_EQ(rm, key);

    Cblacs_gridexit(context);
  }
}
#endif

TEST(GridTest, GridDLAFOrdering) {
  for (const auto& [key, value] : ordering) {
    dlaf::comm::Communicator world(MPI_COMM_WORLD);

    dlaf::comm::CommunicatorGrid row_major(world, 2, 3, value);

    char rm = grid_ordering(world, row_major.size().rows(), row_major.size().cols(),
                            row_major.rank().row(), row_major.rank().col());
    EXPECT_EQ(rm, key);
  }
}
