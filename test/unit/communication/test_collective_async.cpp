//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/communication/kernels/reduce.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/matrix/matrix.h"

TEST(Reduce, Contiguous) {
  using namespace dlaf;
  using namespace std::literals;

  comm::Communicator comm(MPI_COMM_WORLD);
  comm::CommunicatorGrid grid(comm, 1, 2, common::Ordering::ColumnMajor);
  common::Pipeline<comm::Communicator> chain(comm);

  auto ex_mpi = getMPIExecutor<Backend::MC>();

  int root = 0;
  int sz = 4;

  LocalTileIndex index(0, 0);
  dlaf::Matrix<double, Device::CPU> matrix({sz, 1}, {sz, 1});
  matrix(index).get()({0, 0}) = comm.rank() == 0 ? 1 : 3;

  if (comm.rank() == root) {
    auto t = scheduleReduceRecvInPlace(ex_mpi, chain(), MPI_SUM, matrix(index));
    EXPECT_EQ(4, t.get()({0, 0}));
  }
  else {
    auto t = scheduleReduceSend(ex_mpi, root, chain(), MPI_SUM, matrix.read(index));
    EXPECT_EQ(3, t.get()({0, 0}));
  }
}

TEST(Reduce, NotContiguous) {
  using namespace dlaf;
  using namespace std::literals;

  comm::Communicator comm(MPI_COMM_WORLD);
  comm::CommunicatorGrid grid(comm, 1, 2, common::Ordering::ColumnMajor);
  common::Pipeline<comm::Communicator> chain(comm);

  auto ex_mpi = getMPIExecutor<Backend::MC>();

  int root = 0;
  int sz = 4;

  LocalTileIndex index(0, 0);
  dlaf::Matrix<double, Device::CPU> matrix({1, sz}, {1, sz});
  matrix(index).get()({0, 0}) = comm.rank() == 0 ? 1 : 3;

  if (comm.rank() == root) {
    auto t = scheduleReduceRecvInPlace(ex_mpi, chain(), MPI_SUM, matrix(index));
    EXPECT_EQ(4, t.get()({0, 0}));
  }
  else {
    auto t = scheduleReduceSend(ex_mpi, root, chain(), MPI_SUM, matrix.read(index));
    EXPECT_EQ(3, t.get()({0, 0}));
  }
}
