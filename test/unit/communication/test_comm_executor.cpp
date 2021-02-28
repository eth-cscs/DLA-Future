//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "dlaf/communication/communicator.h"
#include "dlaf/communication/executor.h"

template <dlaf::comm::MPIMech M>
void test_exec() {
  auto comm = dlaf::comm::Communicator(MPI_COMM_WORLD);
  int rank = comm.rank();
  int nprocs = comm.size();
  dlaf::comm::Executor<M> ex("default", comm);

  int size = 1000;
  MPI_Datatype dtype = MPI_DOUBLE;
  std::vector<double> send_buf(static_cast<std::size_t>(size), rank);
  std::vector<double> recv_buf(static_cast<std::size_t>(size));
  int send_rank = (rank + 1) % nprocs;
  int recv_rank = (rank != 0) ? rank - 1 : nprocs - 1;
  int tag = 0;

  auto send_fut = hpx::async(ex, MPI_Isend, send_buf.data(), size, dtype, send_rank, tag);
  auto recv_fut = hpx::async(ex, MPI_Irecv, recv_buf.data(), size, dtype, recv_rank, tag);
  hpx::wait_all(send_fut, recv_fut);

  std::vector<double> expected_recv_buf(static_cast<std::size_t>(size), recv_rank);

  ASSERT_TRUE(expected_recv_buf == recv_buf);
}

TEST(SendRecv, Yielding) {
  test_exec<dlaf::comm::MPIMech::Yielding>();
}

TEST(SendRecv, Polling) {
  hpx::mpi::experimental::enable_user_polling internal_helper("default");
  test_exec<dlaf::comm::MPIMech::Polling>();
}

TEST(SendRecv, Blocking) {
  auto comm = dlaf::comm::Communicator(MPI_COMM_WORLD);
  dlaf::comm::Executor<dlaf::comm::MPIMech::Blocking> ex("mpi", comm);

  int root_rank = 0;
  MPI_Datatype dtype = MPI_DOUBLE;
  int size = 1000;
  double val = 4.2;
  std::vector<double> buf(static_cast<std::size_t>(size), val);

  hpx::async(ex, MPI_Bcast, buf.data(), size, dtype, root_rank).get();

  std::vector<double> expected_buf(static_cast<std::size_t>(size), val);

  //hpx::dataflow(ex, MPI_Bcast, buf.data(), size, dtype, root_rank).get();

  ASSERT_TRUE(expected_buf == buf);
}
