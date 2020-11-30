//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "gtest/gtest.h"
#include "dlaf/communication/communicator_grid.h"

namespace dlaf_test {

using namespace dlaf;
using namespace dlaf::comm;

std::vector<CommunicatorGrid> comm_grids;

class CommunicatorGrid1RankEnvironment : public ::testing::Environment {
  static_assert(NUM_MPI_RANKS == 1, "Exactly 1 rank is required");

public:
  virtual void SetUp() override {
    if (comm_grids.empty()) {
      Communicator world(MPI_COMM_WORLD);
      comm_grids.emplace_back(world, 1, 1, common::Ordering::RowMajor);
      comm_grids.emplace_back(world, 1, 1, common::Ordering::ColumnMajor);

      int rows = 1;
      int cols = 1;
      int color = 1;
      MPI_Comm split_comm;
      MPI_Comm_split(world, color, world.rank(), &split_comm);

      Communicator comm(split_comm);
      comm_grids.emplace_back(comm, rows, cols, common::Ordering::ColumnMajor);
    }
  }

  virtual void TearDown() override {
    comm_grids.clear();
  }
};

}
