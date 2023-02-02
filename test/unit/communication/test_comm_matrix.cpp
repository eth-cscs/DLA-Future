//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <pika/future.hpp>

#include <gtest/gtest.h>
#include <mpi.h>

#include "dlaf/communication/communicator.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/sender/transform_mpi.h"

using pika::execution::experimental::just;
using pika::execution::experimental::start_detached;
using pika::execution::experimental::when_all;
using pika::this_thread::experimental::sync_wait;

using namespace dlaf;

TEST(BcastMatrixTest, TransformMPIRW) {
  using namespace std::literals;

  comm::Communicator comm(MPI_COMM_WORLD);
  comm::CommunicatorGrid grid(comm, 1, 2, common::Ordering::ColumnMajor);
  common::Pipeline<comm::Communicator> ccomm(comm);

  int root = 0;
  int sz = 10000;

  LocalTileIndex index(0, 0);
  dlaf::Matrix<double, Device::CPU> mat({sz, 1}, {sz, 1});
  if (comm.rank() == root) {
    sync_wait(mat.readwrite_sender_tile(index))({sz - 1, 0}) = 1.;
    start_detached(when_all(ccomm(), mat.readwrite_sender_tile(index)) |
                   comm::internal::transformMPI(comm::internal::sendBcast_o));
    mat.readwrite_sender_tile(index) |
        transformDetach(internal::Policy<Backend::MC>(), [sz](matrix::Tile<double, Device::CPU> tile) {
          tile({sz - 1, 0}) = 2.;
        });
    EXPECT_EQ(2., sync_wait(mat.read_sender2(index)).get()({sz - 1, 0}));
  }
  else {
    std::this_thread::sleep_for(50ms);
    start_detached(when_all(ccomm(), just(root), mat.readwrite_sender_tile(index)) |
                   comm::internal::transformMPI(comm::internal::recvBcast_o));
    EXPECT_EQ(1., sync_wait(mat.read_sender2(index)).get()({sz - 1, 0}));
  }
}

TEST(BcastMatrixTest, TransformMPIRO) {
  using namespace std::literals;
  using pika::execution::experimental::start_detached;
  using pika::execution::experimental::when_all;

  comm::Communicator comm(MPI_COMM_WORLD);
  comm::CommunicatorGrid grid(comm, 1, 2, common::Ordering::ColumnMajor);
  common::Pipeline<comm::Communicator> ccomm(comm);

  int root = 0;
  int sz = 10000;

  LocalTileIndex index(0, 0);
  dlaf::Matrix<double, Device::CPU> mat({sz, 1}, {sz, 1});
  if (comm.rank() == root) {
    sync_wait(mat.readwrite_sender_tile(index))({sz - 1, 0}) = 1.;
    start_detached(when_all(ccomm(), mat.read_sender2(index)) |
                   comm::internal::transformMPI(comm::internal::sendBcast_o));
    mat.readwrite_sender_tile(index) |
        transformDetach(internal::Policy<Backend::MC>(), [sz](matrix::Tile<double, Device::CPU> tile) {
          tile({sz - 1, 0}) = 2.;
        });
    EXPECT_EQ(2., sync_wait(mat.read_sender2(index)).get()({sz - 1, 0}));
  }
  else {
    std::this_thread::sleep_for(50ms);
    start_detached(when_all(ccomm(), just(root), mat.readwrite_sender_tile(index)) |
                   comm::internal::transformMPI(comm::internal::recvBcast_o));
    EXPECT_EQ(1., sync_wait(mat.read_sender2(index)).get()({sz - 1, 0}));
  }
}
