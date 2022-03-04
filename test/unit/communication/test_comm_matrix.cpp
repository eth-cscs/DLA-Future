//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <pika/future.hpp>

#include <gtest/gtest.h>
#include <mpi.h>

#include "dlaf/communication/communicator.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/matrix/matrix.h"

using namespace dlaf;

TEST(BcastMatrixTest, DataflowFuture) {
  using namespace std::literals;
  comm::Communicator comm(MPI_COMM_WORLD);
  comm::CommunicatorGrid grid(comm, 1, 2, common::Ordering::ColumnMajor);
  common::Pipeline<comm::Communicator> ccomm(comm);
  comm::Executor ex{};

  int root = 0;
  int sz = 10000;

  LocalTileIndex index(0, 0);
  dlaf::Matrix<double, Device::CPU> mat({sz, 1}, {sz, 1});
  if (comm.rank() == root) {
    mat(index).get()({sz - 1, 0}) = 1.;
    pika::dataflow(ex, matrix::unwrapExtendTiles(comm::sendBcast_o), mat(index), ccomm());
    pika::dataflow(pika::unwrapping([sz](auto tile) { tile({sz - 1, 0}) = 2.; }), mat(index));
    EXPECT_EQ(2., mat.read(index).get()({sz - 1, 0}));
  }
  else {
    std::this_thread::sleep_for(50ms);
    pika::dataflow(ex, matrix::unwrapExtendTiles(comm::recvBcast_o), mat(index), root, ccomm());
    EXPECT_EQ(1., mat.read(index).get()({sz - 1, 0}));
  }
}

TEST(BcastMatrixTest, DataflowSharedFuture) {
  using namespace std::literals;
  comm::Communicator comm(MPI_COMM_WORLD);
  comm::CommunicatorGrid grid(comm, 1, 2, common::Ordering::ColumnMajor);
  common::Pipeline<comm::Communicator> ccomm(comm);
  comm::Executor ex{};

  int root = 0;
  int sz = 10000;

  LocalTileIndex index(0, 0);
  dlaf::Matrix<double, Device::CPU> mat({sz, 1}, {sz, 1});
  if (comm.rank() == root) {
    mat(index).get()({sz - 1, 0}) = 1.;
    pika::dataflow(ex, matrix::unwrapExtendTiles(comm::sendBcast_o), mat.read(index), ccomm());
    pika::dataflow(pika::unwrapping([sz](auto tile) { tile({sz - 1, 0}) = 2.; }), mat(index));
    EXPECT_EQ(2., mat.read(index).get()({sz - 1, 0}));
  }
  else {
    std::this_thread::sleep_for(50ms);
    pika::dataflow(ex, matrix::unwrapExtendTiles(comm::recvBcast_o), mat(index), root, ccomm());
    EXPECT_EQ(1., mat.read(index).get()({sz - 1, 0}));
  }
}
