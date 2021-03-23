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
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "dlaf/communication/communicator.h"
#include "dlaf/communication/executor.h"

void test_exec(dlaf::comm::MPIMech mech) {
  auto comm = dlaf::comm::Communicator(MPI_COMM_WORLD);
  int rank = comm.rank();
  int nprocs = comm.size();
  dlaf::comm::Executor ex("default", mech);

  int size = 1000;
  MPI_Datatype dtype = MPI_DOUBLE;
  std::vector<double> send_buf(static_cast<std::size_t>(size), rank);
  std::vector<double> recv_buf(static_cast<std::size_t>(size));
  int send_rank = (rank + 1) % nprocs;
  int recv_rank = (rank != 0) ? rank - 1 : nprocs - 1;
  int tag = 0;

  auto send_fut = hpx::async(ex, MPI_Isend, send_buf.data(), size, dtype, send_rank, tag, comm);
  auto recv_fut = hpx::async(ex, MPI_Irecv, recv_buf.data(), size, dtype, recv_rank, tag, comm);
  hpx::wait_all(send_fut, recv_fut);

  std::vector<double> expected_recv_buf(static_cast<std::size_t>(size), recv_rank);

  ASSERT_TRUE(expected_recv_buf == recv_buf);
}

TEST(SendRecv, Yielding) {
  test_exec(dlaf::comm::MPIMech::Yielding);
}

TEST(SendRecv, Polling) {
  hpx::mpi::experimental::enable_user_polling internal_helper("default");
  test_exec(dlaf::comm::MPIMech::Polling);
}

TEST(Bcast, Dataflow) {
  auto comm = dlaf::comm::Communicator(MPI_COMM_WORLD);
  dlaf::comm::Executor ex("default", dlaf::comm::MPIMech::Yielding);
  int root_rank = 1;
  MPI_Datatype dtype = MPI_DOUBLE;
  int size = 1000;
  double val = (comm.rank() == root_rank) ? 4.2 : 1.2;
  std::vector<double> buf(static_cast<std::size_t>(size), val);

  // Tests the handling of futures in a dataflow
  hpx::dataflow(ex, hpx::util::unwrapping(MPI_Ibcast), buf.data(), hpx::make_ready_future<int>(size),
                dtype, root_rank, comm, hpx::make_ready_future<void>())
      .get();

  std::vector<double> expected_buf(static_cast<std::size_t>(size), 4.2);
  ASSERT_TRUE(expected_buf == buf);
}
