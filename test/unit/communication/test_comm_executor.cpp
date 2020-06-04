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
#include <hpx/lcos/async.hpp>
#include <hpx/lcos/wait_all.hpp>

#include "dlaf/communication/communicator.h"
#include "dlaf/communication/executor.h"

TEST(Comm, SendRecv) {
  // A send-recv cycle between 4 processes
  // hpx::threads::executors::pool_executor ex("default");

  auto comm = dlaf::comm::Communicator(MPI_COMM_WORLD);
  int rank = comm.rank();
  int nprocs = comm.size();
  dlaf::comm::executor ex(comm);

  int size = 1000;
  MPI_Datatype dtype = MPI_DOUBLE;
  std::vector<double> send_buf(static_cast<std::size_t>(size), rank);
  std::vector<double> recv_buf(static_cast<std::size_t>(size));
  int send_rank = (rank + 1) % nprocs;
  int recv_rank = (rank != 0) ? rank - 1 : nprocs - 1;
  int tag = 0;

  auto send_fut = ex.async_execute(MPI_Isend, send_buf.data(), size, dtype, send_rank, tag);
  auto recv_fut = ex.async_execute(MPI_Irecv, recv_buf.data(), size, dtype, recv_rank, tag);
  hpx::wait_all(send_fut, recv_fut);

  std::vector<double> expected_recv_buf(static_cast<std::size_t>(size), recv_rank);

  ASSERT_TRUE(expected_recv_buf == recv_buf);
}
