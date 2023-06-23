//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <functional>
#include <sstream>
#include <tuple>

#include <pika/init.hpp>
#include <pika/runtime.hpp>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/factorization/cholesky.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf_c/grid.h>
#include <dlaf_c/init.h>

#include <gtest/gtest.h>

#include <dlaf_test/blacs.h>
#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/matrix/util_generic_lapack.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

TEST(GridTest, GridScaLAPACKOrderingR) {
  char order = 'R';

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
  EXPECT_EQ(rm, 'R');

  Cblacs_gridexit(context);
}

TEST(GridTest, GridScaLAPACKOrderingC) {
  char order = 'C';

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
  EXPECT_EQ(rm, 'R');

  Cblacs_gridexit(context);
}

TEST(GridTest, GridDLAFOrderingR) {
  comm::Communicator world(MPI_COMM_WORLD);

  comm::CommunicatorGrid row_major(world, 2, 3, common::Ordering::RowMajor);

  char rm = grid_ordering(world, row_major.size().rows(), row_major.size().cols(),
                          row_major.rank().row(), row_major.rank().col());
  EXPECT_EQ(rm, 'R');
}

TEST(GridTest, GridDLAFOrderingC) {
  comm::Communicator world(MPI_COMM_WORLD);

  comm::CommunicatorGrid col_major(world, 2, 3, common::Ordering::ColumnMajor);

  char cm = grid_ordering(world, col_major.size().rows(), col_major.size().cols(),
                          col_major.rank().row(), col_major.rank().col());
  EXPECT_EQ(cm, 'C');
}
