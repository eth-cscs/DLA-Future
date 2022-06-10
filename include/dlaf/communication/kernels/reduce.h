//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#ifdef DLAF_WITH_CUDA_RDMA
#warning "Reduce is not using CUDA_RDMA."
#endif

/// @file

#include <mpi.h>

#include <pika/execution.hpp>
#include <pika/future.hpp>
#include <pika/unwrap.hpp>

#include "dlaf/common/contiguous_buffer_holder.h"
#include "dlaf/common/data.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"
#include "dlaf/communication/with_contiguous_buffer.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/schedulers.h"
#include "dlaf/sender/traits.h"
#include "dlaf/sender/transform_mpi.h"

namespace dlaf {
namespace comm {
namespace internal {
template <class T, Device D>
void reduceRecvInPlace(common::PromiseGuard<comm::Communicator> pcomm, MPI_Op reduce_op,
                       const matrix::Tile<T, D>& tile, MPI_Request* req) {
  DLAF_ASSERT(tile.is_contiguous(), "");

  auto& comm = pcomm.ref();
  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(MPI_Ireduce(MPI_IN_PLACE, msg.data(), msg.count(), msg.mpi_type(), reduce_op,
                                   comm.rank(), comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(reduceRecvInPlace);

template <class T, Device D>
void reduceSend(comm::IndexT_MPI rank_root, common::PromiseGuard<comm::Communicator> pcomm,
                MPI_Op reduce_op, const matrix::Tile<const T, D>& tile, MPI_Request* req) {
  DLAF_ASSERT(tile.is_contiguous(), "");

  auto& comm = pcomm.ref();
  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(
      MPI_Ireduce(msg.data(), nullptr, msg.count(), msg.mpi_type(), reduce_op, rank_root, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(reduceSend);
}

/// Given a GPU tile, perform MPI_Reduce in-place
template <class CommSender, class T, Device D>
void scheduleReduceRecvInPlace(CommSender&& pcomm, MPI_Op reduce_op,
                               pika::future<matrix::Tile<T, D>> tile) {
  // Note:
  //
  // GPU --> Duplicate --> (cCPU --> MPI --> cCPU) --> copy --> GPU
  //
  // where: cCPU = contiguous CPU

  namespace ex = pika::execution::experimental;

  using dlaf::comm::internal::transformMPI;
  using dlaf::comm::internal::with_similar_contiguous_comm_tile;
  using dlaf::internal::getBackendScheduler;
  using dlaf::internal::Policy;
  using dlaf::internal::whenAllLift;
  using dlaf::matrix::copy;
  using pika::threads::thread_priority;

  ex::start_detached(
      with_similar_contiguous_comm_tile(std::move(tile),
                                        [reduce_op, pcomm = std::forward<CommSender>(
                                                        pcomm)](auto const& tile_in,
                                                                auto const& tile_contig_comm) mutable
                                        -> ex::unique_any_sender<> {
                                          constexpr Device in_device_type =
                                              std::decay_t<decltype(tile_in)>::D;
                                          constexpr Device comm_device_type =
                                              std::decay_t<decltype(tile_contig_comm)>::D;
                                          constexpr Backend copy_backend =
                                              dlaf::matrix::internal::CopyBackend_v<in_device_type,
                                                                                    comm_device_type>;

                                          auto recv_sender = whenAllLift(std::move(pcomm), reduce_op,
                                                                         std::cref(tile_contig_comm)) |
                                                             transformMPI(internal::reduceRecvInPlace_o);

                                          // This is "copy back if needed".
                                          // Separate helper? copyIfNeeded?
                                          // operator== for Tile (the below is
                                          // not 100% accurate if we have
                                          // views)?
                                          if (tile_in.ptr() == tile_contig_comm.ptr()) {
                                            return {std::move(recv_sender)};
                                          }
                                          else {
                                            // Copy the received data from the
                                            // comm tile to the input tile.
                                            return whenAllLift(std::move(recv_sender),
                                                               std::cref(tile_contig_comm),
                                                               std::cref(tile_in)) |
                                                   copy(Policy<copy_backend>(thread_priority::high));
                                          }
                                        }));
}

// TODO scheduleReduceSend with future will require to move the actual value, not the cref
/// Given a GPU tile perform MPI_Reduce in-place
template <class T, Device D, class CommSender>
void scheduleReduceSend(comm::IndexT_MPI rank_root, CommSender&& pcomm, MPI_Op reduce_op,
                        pika::shared_future<matrix::Tile<const T, D>> tile) {
  // Note:
  //
  // GPU --> Duplicate --> (cCPU --> MPI --> cCPU)
  //
  // where: cCPU = contiguous CPU

  namespace ex = pika::execution::experimental;

  using dlaf::comm::internal::transformMPI;
  using dlaf::comm::internal::with_contiguous_comm_tile;
  using dlaf::internal::getBackendScheduler;
  using dlaf::internal::whenAllLift;

  ex::start_detached(
      with_contiguous_comm_tile(ex::keep_future(std::move(tile)),
                                [rank_root, reduce_op,
                                 pcomm = std::forward<CommSender>(
                                     pcomm)](auto const&, auto const& tile_contig_comm) mutable {
                                  return whenAllLift(rank_root, std::move(pcomm), reduce_op,
                                                     std::cref(tile_contig_comm)) |
                                         transformMPI(internal::reduceSend_o);
                                }));
}
}
}
