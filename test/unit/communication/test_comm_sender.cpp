//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <pika/execution.hpp>
#include <pika/future.hpp>
#include <pika/mpi.hpp>

#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "dlaf/communication/communicator.h"
#include "dlaf/communication/error.h"

using pika::execution::experimental::just;
using pika::execution::experimental::then;
using pika::execution::experimental::when_all;
using pika::mpi::experimental::transform_mpi;
using pika::this_thread::experimental::sync_wait;

void test_transform_mpi() {
  auto comm = dlaf::comm::Communicator(MPI_COMM_WORLD);
  int rank = comm.rank();
  int nprocs = comm.size();

  int size = 1000;
  MPI_Datatype dtype = MPI_DOUBLE;
  std::vector<double> send_buf(static_cast<std::size_t>(size), rank);
  std::vector<double> recv_buf(static_cast<std::size_t>(size));
  int send_rank = (rank + 1) % nprocs;
  int recv_rank = (rank != 0) ? rank - 1 : nprocs - 1;
  int tag = 0;

  auto send = just(send_buf.data(), size, dtype, send_rank, tag, comm) | transform_mpi(MPI_Isend);
  auto recv = just(recv_buf.data(), size, dtype, recv_rank, tag, comm) | transform_mpi(MPI_Irecv);
  when_all(std::move(send), std::move(recv)) | then([](int e1, int e2) {
    DLAF_MPI_CHECK_ERROR(e1);
    DLAF_MPI_CHECK_ERROR(e2);
  }) | sync_wait();

  std::vector<double> expected_recv_buf(static_cast<std::size_t>(size), recv_rank);

  ASSERT_TRUE(expected_recv_buf == recv_buf);
}

TEST(SendRecv, Polling) {
  test_transform_mpi();
}

TEST(Bcast, Polling) {
  auto comm = dlaf::comm::Communicator(MPI_COMM_WORLD);
  int root_rank = 1;
  MPI_Datatype dtype = MPI_DOUBLE;
  int size = 1000;
  double val = (comm.rank() == root_rank) ? 4.2 : 1.2;
  std::vector<double> buf(static_cast<std::size_t>(size), val);

  when_all(just(buf.data()), pika::make_ready_future<int>(size), just(dtype, root_rank, comm),
           pika::make_ready_future<void>()) |
      transform_mpi(MPI_Ibcast) | then([](int e) { DLAF_MPI_CHECK_ERROR(e); }) | sync_wait();

  std::vector<double> expected_buf(static_cast<std::size_t>(size), 4.2);
  ASSERT_TRUE(expected_buf == buf);
}
