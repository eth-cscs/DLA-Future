//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "dlaf/communication/communicator.h"
#include "dlaf/communication/error.h"

namespace dlaf {
namespace comm {
namespace test {

/// Test fixture that split even/odd ranks (wrt rank id) in two separate communicators.
class SplittedCommunicatorsTest : public ::testing::Test {
  static_assert(NUM_MPI_RANKS >= 4, "At least 4 ranks, in order to avoid single rank communicators");

protected:
  void SetUp() override {
    world = Communicator(MPI_COMM_WORLD);

    color = world.rank() % 2;
    key = world.rank() / 2;

    MPI_Comm mpi_splitted_comm;
    DLAF_MPI_CHECK_ERROR(MPI_Comm_split(world, color, key, &mpi_splitted_comm));

    ASSERT_NE(MPI_COMM_NULL, mpi_splitted_comm);
    splitted_comm = Communicator(mpi_splitted_comm);
  }

  void TearDown() override {
    if (MPI_COMM_NULL != splitted_comm)
      DLAF_MPI_CHECK_ERROR(MPI_Comm_free(&splitted_comm));
  }

  Communicator world;          ///< the world communicator
  Communicator splitted_comm;  ///< even/odd communicator, based on what category this rank belongs to

  int color = MPI_UNDEFINED;  ///< color tells the category (even/odd) of the current rank
  int key = MPI_UNDEFINED;    ///< key is the rank id in @p splitted_comm
};

}
}
}
