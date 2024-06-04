//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cstddef>
#include <utility>
#include <vector>

#include <mpi.h>

#include <pika/execution.hpp>

#include <dlaf/communication/communicator.h>
#include <dlaf/communication/error.h>
#include <dlaf/sender/transform_mpi.h>

#include <gtest/gtest.h>

using dlaf::comm::internal::transformMPI;
using pika::execution::experimental::just;
using pika::execution::experimental::then;
using pika::execution::experimental::when_all;
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

  auto send = just(send_buf.data(), size, dtype, send_rank, tag, comm) | transformMPI(MPI_Isend);
  auto recv = just(recv_buf.data(), size, dtype, recv_rank, tag, comm) | transformMPI(MPI_Irecv);
  sync_wait(when_all(std::move(send), std::move(recv)) | then([]() {}));

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

  sync_wait(just(buf.data(), size, dtype, root_rank, comm) | transformMPI(MPI_Ibcast) | then([]() {}));

  std::vector<double> expected_buf(static_cast<std::size_t>(size), 4.2);
  ASSERT_TRUE(expected_buf == buf);
}
