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

#include <pika/unwrap.hpp>

#include "dlaf/common/contiguous_buffer_holder.h"
#include "dlaf/common/data.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/message.h"
#include "dlaf/matrix/tile.h"

namespace dlaf {
namespace comm {

namespace internal {

template <class T>
void reduceRecvInPlace(const common::PromiseGuard<comm::Communicator>& pcomm, MPI_Op reduce_op,
                       const common::internal::ContiguousBufferHolder<T>& cont_buf, MPI_Request* req) {
  auto msg = comm::make_message(cont_buf.descriptor);
  auto& comm = pcomm.ref();

  DLAF_MPI_CALL(MPI_Ireduce(MPI_IN_PLACE, msg.data(), msg.count(), msg.mpi_type(), reduce_op,
                            comm.rank(), comm, req));
}

template <class T>
void reduceSend(comm::IndexT_MPI rank_root, const common::PromiseGuard<comm::Communicator>& pcomm,
                MPI_Op reduce_op, const common::internal::ContiguousBufferHolder<const T>& cont_buf,
                MPI_Request* req) {
  auto msg = comm::make_message(cont_buf.descriptor);
  auto& comm = pcomm.ref();

  DLAF_MPI_CALL(
      MPI_Ireduce(msg.data(), nullptr, msg.count(), msg.mpi_type(), reduce_op, rank_root, comm, req));
}

// TODO SenderElementType of a reference_wrapper does not work
template <class T, class Sender>
auto senderReduceRecvInPlace(const comm::Executor& ex,
                             pika::future<common::PromiseGuard<comm::Communicator>> pcomm,
                             MPI_Op reduce_op, Sender&& tile) {
  // Note:
  //
  // CPU  ---> makeItContiguous ---> cCPU -> MPI ---> copyBack ---> CPU
  // cCPU -------------------------> cCPU -> MPI -----------------> cCPU
  //
  // where: cCPU = contiguous CPU

  using namespace pika::execution::experimental;

  using dlaf::matrix::Tile;
  using dlaf::internal::Policy;
  using dlaf::internal::transform;
  using dlaf::internal::whenAllLift;

  return std::forward<Sender>(tile) | transfer(thread_pool_scheduler{}) |
         let_value([ex, reduce_op, pcomm = std::move(pcomm)](Tile<T, Device::CPU>& tile_orig) mutable {
           using dlaf::common::internal::makeItContiguous;

           auto contiguous_buffer = makeItContiguous(tile_orig);
           return just(std::move(contiguous_buffer)) |
                  let_value([ex, reduce_op, &tile_orig,
                             pcomm = std::move(pcomm)](const auto& cont_buffer) mutable {
                    return pika::dataflow(ex, internal::reduceRecvInPlace<T>, pcomm.get(), reduce_op,
                                          std::cref(cont_buffer)) |
                           then([&]() {
                             // TODO ref_wrapper to matrix::Tile const& is not converted
                             dlaf::common::internal::copyBack(cont_buffer, tile_orig);
                             // TODO plus I want to return the original tile
                             return std::move(tile_orig);
                           });
                  });
         });
}

template <class T, class Sender>
auto senderReduceSend(const comm::Executor& ex, comm::IndexT_MPI rank_root,
                      pika::future<common::PromiseGuard<comm::Communicator>> pcomm, MPI_Op reduce_op,
                      Sender&& tile) {
  // Note:
  //
  // CPU  ---> makeItContiguous ---> cCPU -> MPI
  // cCPU -------------------------> cCPU -> MPI
  //
  // where: cCPU = contiguous CPU

  using namespace pika::execution::experimental;
  using dlaf::internal::whenAllLift;

  return std::forward<Sender>(tile) |  // TODO transfer on CPU?
         let_value([ex, rank_root, reduce_op,
                    pcomm = std::move(pcomm)](const matrix::Tile<const T, Device::CPU>& tile) mutable {
           using common::internal::makeItContiguous;
           return whenAllLift(std::move(pcomm), makeItContiguous(tile)) |
                  let_value([ex, rank_root, reduce_op](auto& pcomm, const auto& cont_buf) {
                    return pika::dataflow(ex, internal::reduceSend<T>, rank_root, std::move(pcomm),
                                          reduce_op, std::cref(cont_buf));
                  });
         });
}
}

template <class T>
void scheduleReduceRecvInPlace(const comm::Executor& ex,
                               pika::future<common::PromiseGuard<comm::Communicator>> pcomm,
                               MPI_Op reduce_op, pika::future<matrix::Tile<T, Device::CPU>> tile) {
  using namespace pika::execution::experimental;

  internal::senderReduceRecvInPlace<T>(ex, std::move(pcomm), reduce_op, std::move(tile)) |
      start_detached();
}

template <class T, Device D>
void scheduleReduceRecvInPlace(const comm::Executor& ex,
                               pika::future<common::PromiseGuard<comm::Communicator>> pcomm,
                               MPI_Op reduce_op, pika::future<matrix::Tile<T, D>> tile) {
  // Note:
  //
  // GPU ---> Duplicate ---> (cCPU --> MPI) ---> copy --> GPU
  //
  // where: cCPU = contiguous CPU

  using namespace pika::execution::experimental;
  using dlaf::internal::transform;

  std::move(tile) |
      let_value(pika::unwrapping([ex, reduce_op, pcomm = std::move(pcomm)](auto& tile_gpu) mutable {
        using dlaf::internal::Policy;
        using dlaf::matrix::internal::CopyBackend;

        auto tile_cpu = transform(
            Policy<CopyBackend<D, Device::CPU>::value>(pika::threads::thread_priority::high),
            [](const matrix::Tile<const T, Device::GPU>& tile_gpu, auto... args) mutable {
              return dlaf::matrix::Duplicate<Device::CPU>{}(tile_gpu, args...);
            },
            just(std::cref(tile_gpu)));
        auto tile_reduced =
            internal::senderReduceRecvInPlace<T>(ex, std::move(pcomm), reduce_op, std::move(tile_cpu));
        // TODO matrix::copy(Policy<Backend::GPU>(pika::threads::thread_priority::high));
        return transform(
            Policy<CopyBackend<Device::CPU, D>::value>(pika::threads::thread_priority::high),
            [&](const matrix::Tile<T, Device::CPU>& tile_cpu, auto... args) {
              matrix::internal::copy(tile_cpu, tile_gpu, args...);
            },
            std::move(tile_reduced));
      })) |
      start_detached();
}

// TODO scheduleReduceSend with future will require to move the actual value, not the cref
template <class T>
void scheduleReduceSend(const comm::Executor& ex, comm::IndexT_MPI rank_root,
                        pika::future<common::PromiseGuard<comm::Communicator>> pcomm, MPI_Op reduce_op,
                        pika::shared_future<matrix::Tile<const T, Device::CPU>> tile) {
  using namespace pika::execution::experimental;

  return keep_future(std::move(tile)) |
         let_value(pika::unwrapping([ex, rank_root, reduce_op, pcomm = std::move(pcomm)](
                                       const matrix::Tile<const T, Device::CPU>& tile) mutable {
           return internal::senderReduceSend<T>(ex, rank_root, std::move(pcomm), reduce_op,
                                                just(std::cref(tile)));
         })) |
         start_detached();
}

// TODO scheduleReduceSend with future will require to move the actual value, not the cref
template <class T, Device D>
void scheduleReduceSend(const comm::Executor& ex, comm::IndexT_MPI rank_root,
                        pika::future<common::PromiseGuard<comm::Communicator>> pcomm, MPI_Op reduce_op,
                        pika::shared_future<matrix::Tile<const T, D>> tile) {
  // Note:
  //
  // GPU ---> Duplicate ---> (cCPU --> MPI)
  //
  // where: cCPU = contiguous CPU

  using namespace pika::execution::experimental;
  using dlaf::internal::Policy;
  using dlaf::matrix::internal::CopyBackend;

  auto tile_cpu = dlaf::internal::transform(
      Policy<CopyBackend<D, Device::CPU>::value>(pika::threads::thread_priority::high),
      [](const matrix::Tile<const T, Device::GPU>& tile_gpu, auto... args) {
        return dlaf::matrix::Duplicate<Device::CPU>{}(tile_gpu, args...);
      },
      keep_future(std::move(tile)));

  internal::senderReduceSend<T>(ex, rank_root, std::move(pcomm), reduce_op, std::move(tile_cpu)) |
      start_detached();
}
}
}
