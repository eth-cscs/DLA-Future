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
#include <pika/mpi.hpp>
#include <pika/unwrap.hpp>

#include "dlaf/common/contiguous_buffer_holder.h"
#include "dlaf/common/data.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"
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

template <class T>
void reduceSend(comm::IndexT_MPI rank_root, common::PromiseGuard<comm::Communicator> pcomm,
                MPI_Op reduce_op, const common::internal::ContiguousBufferHolder<const T>& cont_buf,
                MPI_Request* req) {
  auto msg = comm::make_message(cont_buf.descriptor);
  auto& comm = pcomm.ref();

  DLAF_MPI_CHECK_ERROR(
      MPI_Ireduce(msg.data(), nullptr, msg.count(), msg.mpi_type(), reduce_op, rank_root, comm, req));
}

template <class Sender>
auto senderReduceRecvInPlace(pika::future<common::PromiseGuard<comm::Communicator>> pcomm,
                             MPI_Op reduce_op, Sender&& tile) {
  // Note:
  //
  // CPU  ---> makeItContiguous ---> cCPU -> MPI ---> copyBack ---> CPU
  // cCPU -------------------------> cCPU -> MPI -----------------> cCPU
  //
  // where: cCPU = contiguous CPU

  using namespace pika::execution::experimental;
  using namespace pika::mpi::experimental;

  using dlaf::internal::getBackendScheduler;
  using dlaf::internal::Policy;
  using dlaf::internal::transform;
  using dlaf::internal::whenAllLift;
  using dlaf::matrix::Tile;

  using T = dlaf::internal::SenderElementType<Sender>;

  return std::forward<Sender>(tile) | transfer(getBackendScheduler<Backend::MC>()) |
         let_value([reduce_op, pcomm = std::move(pcomm)](Tile<T, Device::CPU>& tile_orig) mutable {
           using dlaf::common::internal::makeItContiguous;

           return just(makeItContiguous(tile_orig)) |
                  let_value([reduce_op, &tile_orig,
                             pcomm = std::move(pcomm)](const auto& cont_buffer) mutable {
                    return whenAllLift(pcomm.get(), reduce_op, std::cref(cont_buffer)) |
                           transformMPI(internal::reduceRecvInPlace<T>) | then([&]() {
                             // note: this lambda does two things:
                             //       - avoid implicit conversion problem from reference_wrapper
                             //       - act as if the copy returns the destination tile
                             dlaf::common::internal::copyBack(cont_buffer, tile_orig);
                             return std::move(tile_orig);
                           });
                  });
         });
}

template <class Sender>
auto senderReduceSend(comm::IndexT_MPI rank_root,
                      pika::future<common::PromiseGuard<comm::Communicator>> pcomm, MPI_Op reduce_op,
                      Sender&& tile) {
  // Note:
  //
  // CPU  ---> makeItContiguous ---> cCPU -> MPI
  // cCPU -------------------------> cCPU -> MPI
  //
  // where: cCPU = contiguous CPU

  using namespace pika::execution::experimental;
  using namespace pika::mpi::experimental;
  using dlaf::internal::getBackendScheduler;
  using dlaf::internal::whenAllLift;

  using T = dlaf::internal::SenderElementType<Sender>;

  return std::forward<Sender>(tile) | transfer(getBackendScheduler<Backend::MC>()) |
         let_value(pika::unwrapping([rank_root, reduce_op, pcomm = std::move(pcomm)](
                                        const matrix::Tile<const T, Device::CPU>& tile) mutable {
           using common::internal::makeItContiguous;
           return whenAllLift(std::move(pcomm), makeItContiguous(tile)) |
                  let_value([rank_root, reduce_op](auto& pcomm, const auto& cont_buf) {
                    return whenAllLift(rank_root, std::move(pcomm), reduce_op, std::cref(cont_buf)) |
                           transformMPI(internal::reduceSend<T>);
                  });
         }));
}
}

/// Given a CPU tile, contiguous or not, perform MPI_Reduce in-place
template <class T>
void scheduleReduceRecvInPlace(pika::future<common::PromiseGuard<comm::Communicator>> pcomm,
                               MPI_Op reduce_op, pika::future<matrix::Tile<T, Device::CPU>> tile) {
  // Note:
  //
  // (CPU/cCPU) --> MPI --> (CPU/cCPU)
  //
  // where: cCPU = contiguous CPU

  using namespace pika::execution::experimental;

  internal::senderReduceRecvInPlace(std::move(pcomm), reduce_op, std::move(tile)) | start_detached();
}

/// Given a GPU tile, perform MPI_Reduce in-place
template <class T, Device D>
void scheduleReduceRecvInPlace(pika::future<common::PromiseGuard<comm::Communicator>> pcomm,
                               MPI_Op reduce_op, pika::future<matrix::Tile<T, D>> tile) {
  // Note:
  //
  // GPU --> Duplicate --> (cCPU --> MPI --> cCPU) --> copy --> GPU
  //
  // where: cCPU = contiguous CPU

  using namespace pika::execution::experimental;
  using dlaf::internal::transform;

  std::move(tile) |
      let_value(pika::unwrapping([reduce_op, pcomm = std::move(pcomm)](auto& tile_gpu) mutable {
        using dlaf::internal::Policy;
        using dlaf::matrix::internal::CopyBackend;

        // GPU -> cCPU
        auto tile_cpu = transform(
            Policy<CopyBackend<D, Device::CPU>::value>(pika::threads::thread_priority::high),
            [](const matrix::Tile<const T, Device::GPU>& tile_gpu, auto... args) mutable {
              return dlaf::matrix::Duplicate<Device::CPU>{}(tile_gpu, args...);
            },
            just(std::cref(tile_gpu)));

        // cCPU -> MPI -> cCPU
        auto tile_reduced =
            internal::senderReduceRecvInPlace(std::move(pcomm), reduce_op, std::move(tile_cpu));

        // cCPU -> GPU
        namespace arg = std::placeholders;
        return transform(Policy<CopyBackend<Device::CPU, D>::value>(
                             pika::threads::thread_priority::high),
                         std::bind(matrix::internal::copy_o, arg::_1, std::cref(tile_gpu), arg::_2),
                         std::move(tile_reduced));
      })) |
      start_detached();
}

// TODO scheduleReduceSend with future will require to move the actual value, not the cref
/// Given a CPU tile, being it contiguous or not, perform MPI_Reduce in-place
template <class T>
void scheduleReduceSend(comm::IndexT_MPI rank_root,
                        pika::future<common::PromiseGuard<comm::Communicator>> pcomm, MPI_Op reduce_op,
                        pika::shared_future<matrix::Tile<const T, Device::CPU>> tile) {
  // Note:
  //
  // (CPU/cCPU) --> MPI --> (CPU/cCPU)
  //
  // where: cCPU = contiguous CPU

  using namespace pika::execution::experimental;

  internal::senderReduceSend(rank_root, std::move(pcomm), reduce_op, keep_future(std::move(tile))) |
      start_detached();
}

// TODO scheduleReduceSend with future will require to move the actual value, not the cref
/// Given a GPU tile perform MPI_Reduce in-place
template <class T, Device D>
void scheduleReduceSend(comm::IndexT_MPI rank_root,
                        pika::future<common::PromiseGuard<comm::Communicator>> pcomm, MPI_Op reduce_op,
                        pika::shared_future<matrix::Tile<const T, D>> tile) {
  // Note:
  //
  // GPU --> Duplicate --> (cCPU --> MPI --> cCPU)
  //
  // where: cCPU = contiguous CPU

  using namespace pika::execution::experimental;
  using dlaf::internal::Policy;
  using dlaf::matrix::internal::CopyBackend;

  auto tile_cpu =
      dlaf::internal::transform(Policy<CopyBackend<D, Device::CPU>::value>(
                                    pika::threads::thread_priority::high),
                                dlaf::matrix::Duplicate<Device::CPU>{}, keep_future(std::move(tile)));

  internal::senderReduceSend(rank_root, std::move(pcomm), reduce_op, std::move(tile_cpu)) |
      start_detached();
}
}
}
