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

// TEST(GridTest, GridScaLAPACKOrderingR) {
//
//
//     char rm = grid_ordering(row_major.fullCommunicator(), row_major.size().rows(), row_major.size().cols(), row_major.rank().row(), row_major.rank().col()); 
//     EXPECT_EQ(rm, 'R');
// }
//
// TEST(GridTest, GridScaLAPACKOrderingC) {
//
//     comm::CommunicatorGrid col_major(world, 2, 3, common::Ordering::ColumnMajor);
//     
//     std::cout << "DEBUG " << col_major.rank() << ' ' << world.rank() << ' ' << col_major.size() << std::endl;
//     
//     char cm = grid_ordering(col_major.fullCommunicator(), col_major.size().rows(), col_major.size().cols(), col_major.rank().row(), col_major.rank().col()); 
//     EXPECT_EQ(cm, 'C');
// }

TEST(GridTest, GridDLAFOrderingR) {
    comm::Communicator world(MPI_COMM_WORLD);

    comm::CommunicatorGrid row_major(world, 2, 3, common::Ordering::RowMajor);
    
    char rm = grid_ordering(world, row_major.size().rows(), row_major.size().cols(), row_major.rank().row(), row_major.rank().col()); 
    EXPECT_EQ(rm, 'R');
}

TEST(GridTest, GridDLAFOrderingC) {
    comm::Communicator world(MPI_COMM_WORLD);

    comm::CommunicatorGrid col_major(world, 2, 3, common::Ordering::ColumnMajor);
    
    char cm = grid_ordering(world, col_major.size().rows(), col_major.size().cols(), col_major.rank().row(), col_major.rank().col()); 
    EXPECT_EQ(cm, 'C');
}
