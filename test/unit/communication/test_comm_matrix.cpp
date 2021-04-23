//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <hpx/futures/future.hpp>

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

  if (comm.rank() == root) {
    LocalTileIndex index(0, 0);
    dlaf::Matrix<double, Device::CPU> mat({sz, 1}, {sz, 1});
    mat(index).get()({sz - 1, 0}) = 1.;
    hpx::dataflow(ex, hpx::util::unwrapping(comm::sendBcast<double>), mat(index), ccomm());
    hpx::dataflow(hpx::util::unwrapping([sz](auto tile) { tile({sz - 1, 0}) = 2.; }), mat(index));
    EXPECT_EQ(2., mat(index).get()({sz - 1, 0}));
  }
  else {
    std::this_thread::sleep_for(50ms);
    auto tile_f = hpx::dataflow(ex, hpx::util::unwrapping(comm::recvBcastAlloc<double>),
                                TileElementSize{sz, 1}, root, ccomm());
    EXPECT_EQ(1., tile_f.get()({sz - 1, 0}));
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

  if (comm.rank() == root) {
    LocalTileIndex index(0, 0);
    dlaf::Matrix<double, Device::CPU> mat({sz, 1}, {sz, 1});
    mat(index).get()({sz - 1, 0}) = 1.;
    hpx::dataflow(ex, hpx::util::unwrapping(comm::sendBcast<double>), mat.read(index), ccomm());
    hpx::dataflow(hpx::util::unwrapping([sz](auto tile) { tile({sz - 1, 0}) = 2.; }), mat(index));
    EXPECT_EQ(2., mat(index).get()({sz - 1, 0}));
  }
  else {
    std::this_thread::sleep_for(50ms);
    auto tile_f = hpx::dataflow(ex, hpx::util::unwrapping(comm::recvBcastAlloc<double>),
                                TileElementSize{sz, 1}, root, ccomm());
    EXPECT_EQ(1., tile_f.get()({sz - 1, 0}));
  }
}
