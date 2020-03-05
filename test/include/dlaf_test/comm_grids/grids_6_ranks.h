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

namespace dlaf {
namespace test {

std::vector<comm::CommunicatorGrid> comm_grids;

class CommunicatorGrid6RanksEnvironment : public ::testing::Environment {
  static_assert(NUM_MPI_RANKS == 6, "Exactly 6 ranks are required");
public:
  virtual void SetUp() override {
    if (comm_grids.empty()) {
      comm::Communicator world(MPI_COMM_WORLD);
      comm_grids.emplace_back(world, 3, 2, common::Ordering::RowMajor);
      comm_grids.emplace_back(world, 2, 3, common::Ordering::ColumnMajor);

      int rows = -1;
      int cols = -1;
      int color = -1;
      if (world.rank() < 3) {
        rows = 3;
        cols = 1;
        color = 1;
      }
      else if (world.rank() < 5) {
        rows = 1;
        cols = 2;
        color = 2;
      }
      else {
        rows = 1;
        cols = 1;
        color = 3;
      }
      MPI_Comm split_comm;
      MPI_Comm_split(world, color, world.rank(), &split_comm);

      comm::Communicator comm(split_comm);
      comm_grids.emplace_back(comm, rows, cols, common::Ordering::ColumnMajor);
    }
  }

  virtual void TearDown() override {
    comm_grids.clear();
  }
};

}
}

namespace dlaf_test {
// TODO: remove when cleaning-up namespaces.
using dlaf::test::CommunicatorGrid6RanksEnvironment;
std::vector<dlaf::comm::CommunicatorGrid>& comm_grids = dlaf::test::comm_grids;
}
