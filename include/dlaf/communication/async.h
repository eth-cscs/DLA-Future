//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

/// @file

#include <utility>

#include <mpi.h>
#include <hpx/async.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/util/yield_while.hpp>
#ifdef DLAF_WITH_MPI_FUTURES
#include <hpx/mpi/mpi_executor.hpp>
#include <hpx/mpi/mpi_future.hpp>
#endif

namespace dlaf {
namespace comm {

/// The function only accepts non-blocking MPI routines. The MPI library needs
/// to support MPI_THREAD_MULTIPLE.
///
/// Example usage:
///
/// dlaf::comm::async(ex, comm, MPI_Irecv, recv_ptr, num_recv_elems, MPI_DOUBLE, src_rank, tag);
///
/// compared to the usual MPI non-blocking call:
///
/// MPI_Irecv(recv_ptr, num_recv_elems, MPI_DOUBLE, src_rank, tag, comm, &request);
///
/// there are two differences
///
/// 1) MPI_Comm (comm) goes after the executor in async()
/// 2) MPI_Request (request) is omitted as it is handled internally in async()
///
/// Note that if starting the communication depends on a previous task, it is often a good idea to call
/// the `get()` method on a returned future to avoid weird constructs like future<future<>>.
///
/// Example:
///
/// auto comm_fut = prev_fut.then([=] {
///   return dlaf::comm::async(...).get();
/// });
///
/// or don't return the future from `dlaf::comm::async()` and just do:
///
/// auto comm_fut = prev_fut.then([=] {
///   dlaf::comm::async(...);
/// });
///
template <class F, class... Ts>
hpx::future<void> async(hpx::threads::executors::pool_executor& ex, MPI_Comm comm, F&& f,
                        Ts&&... ts) noexcept {
  return hpx::async(ex, [=] {
    MPI_Request req;
    f(ts..., comm, &req);

    // Yield until non-blocking communication completes.
    hpx::util::yield_while([&req] {
      int flag;
      MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
      return flag == 0;
    });
  });

/// This section is in preparation for MPI_Futures in HPX 1.5. The mechanism for handling MPI
/// communication is similar but there are notable differences.
///
/// Example usage:
///
/// ```
/// // registers a `poll()` function on the thread pool with `pool_name`. This has to be in scope of all
/// // uses of hpx::mpi.
/// hpx::mpi::experimental::enable_user_polling enable_polling(pool_name);
/// ...
/// hpx::mpi::experimental::executor ex(mpi_comm);
/// hpx::async(ex, MPI_Irecv, recv_ptr, num_recv_elems, MPI_DOUBLE, src_rank, tag)
/// ```
///
/// The executor is quite simple and just forwards `MPI_Comm` to the internal
/// hpx::mpi::experimental::detail::async()` call. Note that this is the main API difference with
/// `dlaf::comm::async` where MPI_Comm is not part of the executor (i.e. pool_executor). The only other
/// API difference is that `hpx::async` returns hpx::future<int> where `int` is the error code returned
/// by the MPI call.
///
/// The internal mechanism for handling MPI requests is quite different. All requests are placed into a
/// global array and polled for completion with `MPI_Testany()` on task completion/yielding for all tasks
/// for which polling is enabled (note `enable_user_polling()`).
///
#ifdef DLAF_WITH_MPI_FUTURES
// TODO
#endif
}

}
}
