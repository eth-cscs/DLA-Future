//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/sender/transform_mpi.h"

#include <atomic>
#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/sender/when_all_lift.h"

using namespace dlaf;
using namespace dlaf::comm;

using TransformMPITest = ::testing::Test;

// wait for guard to become true with a timeout of 2000ms
auto try_waiting_guard = [](std::atomic_bool& guard) {
  using namespace std::chrono_literals;
  const auto wait_guard = 20ms;

  for (int i = 0; i < 100 && !guard; ++i)
    std::this_thread::sleep_for(wait_guard);
};

TEST_F(TransformMPITest, PromiseGuardManagement) {
  static_assert(NUM_MPI_RANKS >= 2, "This test requires 2 ranks");

  // Note:
  // Rank 0 is where the test actually is done, while Rank 1 is just used by Rank 0 to control
  // its own progress.
  //
  // Rank 0 posts an IRecv that will be satisfied by Rank 1 just after it receives from Rank 0
  // the trigger. So, actually, Rank 0 can control when to unlock its IRecv by deciding when to
  // send the trigger to Rank 1. In this way, Rank 0 flow can be "paused" for verifiying various
  // phases of the async communication mechanism for IRecv.

  Communicator world(MPI_COMM_WORLD);
  const IndexT_MPI rank = world.rank();

  if (rank == 0) {
    using dlaf::comm::internal::transformMPI;
    using dlaf::internal::whenAllLift;

    namespace ex = pika::execution::experimental;

    common::Pipeline<Communicator> chain(world);

    // Note:
    // `sent_guard` represents the status of completion of IRecv. It will be set true by a `then`
    // task that depends on completion of transformMPI, i.e. it will become true only when IRecv
    // communication completes (i.e. posted + message received).
    std::atomic_bool sent_guard = false;

    int message;
    whenAllLift(&message, 1, MPI_INT, 1, 0, chain()) | transformMPI(MPI_Irecv) |
        ex::then([&sent_guard](auto mpi_err_code) {
          EXPECT_EQ(MPI_SUCCESS, mpi_err_code);
          sent_guard = true;
        }) |
        ex::ensure_started();

    // Note:
    // At this point IRecv is (getting) posted but it won't complete until this Rank 0 will trigger
    // Rank 1 to send the message. So, here we can check that PromiseGuard<Communicator> gets
    // consumed just after MPI operation is posted.
    //
    // For checking it gets released, we ask the Pipeline for the next PromiseGuard<Communicator> in
    // the chain. Indeed, if the previous one is released, this one will be unlocked.
    //
    // Let's use a guard + try_waiting_guard, so that in case something goes wrong it does not end
    // up creating a deadlock. The assumption is that try_waiting_guard timeout is enough for
    // transformMPI to post the IRecv.
    std::atomic_bool pg_guard = false;
    auto after_pg = chain() | ex::then([&pg_guard](auto&& pg) {
                      pg_guard = true;
                      return std::move(pg);
                    }) |
                    ex::ensure_started();
    try_waiting_guard(pg_guard);
    EXPECT_TRUE(pg_guard);

    // "ensure" (by waiting a reasonable amount of time) that IRecv doesn't complete
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_FALSE(sent_guard);

    // Note:
    // At this point we checked if everything went as expected in the posting phase. Let's signal
    // Rank 1 that it can send back the message that will unlock IRecv.
    const int signal = 13;
    MPI_Send(&signal, 1, MPI_INT, 1, 0, world);

    // Note:
    // Here the assumption is that try_waiting_guard timeout is enough for Rank 1 to receive signal,
    // send back the unlocking message, and that on Rank 0 the IRecv gets completed and the MPI
    // scheduler is able to verify its completeness status (i.e. check MPI_Request).
    // Once the transformMPI mechanism completes, the `then` sets the guard signalling the actual
    // full completion of the IRecv via `sent_guard`.
    try_waiting_guard(sent_guard);
    EXPECT_TRUE(sent_guard);

    // this is just checking that the message received by IRecv is actually the one expected.
    EXPECT_EQ(26, message);
  }
  else if (rank == 1) {
    // Note:
    // Rank 1 is just a simple "pinger": as soon as it receives the trigger, it send back a message
    // that acts as signal for Rank 0 IRecv.

    // blocking recv for the trigger
    int buffer;
    MPI_Recv(&buffer, 1, MPI_INT, 0, 0, world, MPI_STATUS_IGNORE);
    EXPECT_EQ(13, buffer);

    // trigger received at this point, so send the actual message that will pair with Rank 0 IRecv
    // allowing it to complete.
    buffer *= 2;
    MPI_Send(&buffer, 1, MPI_INT, 0, 0, world);
  }
}
