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

template <class T>
void reduceRecvInPlace(common::PromiseGuard<comm::Communicator> pcomm, MPI_Op reduce_op,
                       const common::internal::ContiguousBufferHolder<T>& cont_buf, MPI_Request* req) {
  auto msg = comm::make_message(cont_buf.descriptor);
  auto& comm = pcomm.ref();

  DLAF_MPI_CHECK_ERROR(MPI_Ireduce(MPI_IN_PLACE, msg.data(), msg.count(), msg.mpi_type(), reduce_op,
                                   comm.rank(), comm, req));
}

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

template <class CommSender, class Sender>
auto senderReduceRecvInPlace(CommSender&& pcomm, MPI_Op reduce_op, Sender&& tile) {
  // Note:
  //
  // CPU  ---> makeItContiguous ---> cCPU -> MPI ---> copyBack ---> CPU
  // cCPU -------------------------> cCPU -> MPI -----------------> cCPU
  //
  // where: cCPU = contiguous CPU

  namespace ex = pika::execution::experimental;

  using dlaf::internal::getBackendScheduler;
  using dlaf::internal::transform;
  using dlaf::internal::whenAllLift;
  using dlaf::matrix::Tile;

  using T = dlaf::internal::SenderElementType<Sender>;

  return std::forward<Sender>(tile) | ex::transfer(getBackendScheduler<Backend::MC>()) |
         ex::let_value([reduce_op, pcomm = std::forward<CommSender>(pcomm)](
                           Tile<T, Device::CPU>& tile_orig) mutable {
           using dlaf::common::internal::makeItContiguous;

           return ex::just(makeItContiguous(tile_orig)) |
                  ex::let_value([reduce_op, &tile_orig,
                                 pcomm = std::move(pcomm)](const auto& cont_buffer) mutable {
                    return whenAllLift(std::move(pcomm), reduce_op, std::cref(cont_buffer)) |
                           transformMPI(internal::reduceRecvInPlace<T>) | ex::then([&]() {
                             // note: this lambda does two things:
                             //       - avoid implicit conversion problem from reference_wrapper
                             //       - act as if the copy returns the destination tile
                             dlaf::common::internal::copyBack(cont_buffer, tile_orig);
                             return std::move(tile_orig);
                           });
                  });
         });
}
}

/// Given a CPU tile, contiguous or not, perform MPI_Reduce in-place
template <class CommSender, class T>
void scheduleReduceRecvInPlace(CommSender&& pcomm, MPI_Op reduce_op,
                               pika::future<matrix::Tile<T, Device::CPU>> tile) {
  // Note:
  //
  // (CPU/cCPU) --> MPI --> (CPU/cCPU)
  //
  // where: cCPU = contiguous CPU

  using pika::execution::experimental::start_detached;

  internal::senderReduceRecvInPlace(std::forward<CommSender>(pcomm), reduce_op, std::move(tile)) |
      start_detached();
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

  using dlaf::internal::transform;

  std::move(tile) |
      ex::let_value(pika::unwrapping([reduce_op,
                                      pcomm = std::forward<CommSender>(pcomm)](auto& tile_gpu) mutable {
        using dlaf::internal::Policy;
        using dlaf::matrix::copy;
        using dlaf::matrix::internal::CopyBackend;
        using dlaf::internal::whenAllLift;

        // GPU -> cCPU
        auto tile_cpu = transform(
            Policy<CopyBackend<D, Device::CPU>::value>(pika::threads::thread_priority::high),
            [](const matrix::Tile<const T, Device::GPU>& tile_gpu, auto... args) mutable {
              return dlaf::matrix::Duplicate<Device::CPU>{}(tile_gpu, args...);
            },
            ex::just(std::cref(tile_gpu)));

        // cCPU -> MPI -> cCPU
        auto tile_reduced =
            internal::senderReduceRecvInPlace(std::move(pcomm), reduce_op, std::move(tile_cpu));

        // cCPU -> GPU
        namespace arg = std::placeholders;
        return whenAllLift(std::move(tile_reduced), std::cref(tile_gpu)) |
               copy(Policy<CopyBackend<Device::CPU, D>::value>(pika::threads::thread_priority::high));
      })) |
      ex::start_detached();
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
