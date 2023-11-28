//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/error.h>

#include <gtest/gtest.h>

namespace dlaf {
namespace test {

std::vector<comm::CommunicatorGrid> comm_grids;

class CommunicatorGrid6RanksEnvironment : public ::testing::Environment {
  static_assert(NUM_MPI_RANKS == 6, "Exactly 6 ranks are required");

public:
  virtual void SetUp() override {
    if (comm_grids.empty()) {
      comm::Communicator world(MPI_COMM_WORLD);

      // Leave comm_grids empty if invoked with only one rank.
      // Useful to debug local algorithms that otherwise are executed independently on multiple ranks.
      if (world.size() == 1)
        return;

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
      DLAF_MPI_CHECK_ERROR(MPI_Comm_split(world, color, world.rank(), &split_comm));

      auto comm = comm::make_communicator_managed(split_comm);
      comm_grids.emplace_back(comm, rows, cols, common::Ordering::ColumnMajor);
    }
  }

  virtual void TearDown() override {
    comm_grids.clear();
  }
};

class CommunicatorGrid6RanksCAPIEnvironment : public CommunicatorGrid6RanksEnvironment {
public:
  virtual void SetUp() override {
    if (comm_grids.empty()) {
      comm::Communicator world(MPI_COMM_WORLD);

      comm_grids.emplace_back(world, 3, 2, common::Ordering::RowMajor);
      comm_grids.emplace_back(world, 3, 2, common::Ordering::ColumnMajor);
    }
  }
};

struct TestWithCommGrids : public ::testing::Test {
  const std::vector<comm::CommunicatorGrid>& commGrids() {
    EXPECT_FALSE(comm_grids.empty());
    return comm_grids;
  }
};
}
}
