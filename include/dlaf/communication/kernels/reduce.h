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

#include "dlaf/common/callable_object.h"
#include "dlaf/common/contiguous_buffer_holder.h"
#include "dlaf/common/data.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/message.h"
#include "dlaf/communication/rdma.h"
#include "dlaf/executors.h"
#include "dlaf/matrix/tile.h"

namespace dlaf {
namespace comm {

namespace internal {

template <class T>
auto reduceRecvInPlace(common::PromiseGuard<comm::Communicator> pcomm, MPI_Op reduce_op,
                       common::internal::ContiguousBufferHolder<T> cont_buf, MPI_Request* req) {
  auto msg = comm::make_message(cont_buf.descriptor);
  auto& comm = pcomm.ref();

  DLAF_MPI_CALL(MPI_Ireduce(MPI_IN_PLACE, msg.data(), msg.count(), msg.mpi_type(), reduce_op,
                            comm.rank(), comm, req));
  return cont_buf;
}

DLAF_MAKE_CALLABLE_OBJECT(reduceRecvInPlace);

template <class T, Device D>
auto reduceSend(comm::IndexT_MPI rank_root, common::PromiseGuard<comm::Communicator> pcomm,
                MPI_Op reduce_op, common::internal::ContiguousBufferHolder<const T> cont_buf,
                const matrix::Tile<const T, D>&, MPI_Request* req) {
  auto msg = comm::make_message(cont_buf.descriptor);
  auto& comm = pcomm.ref();

  DLAF_MPI_CALL(
      MPI_Ireduce(msg.data(), nullptr, msg.count(), msg.mpi_type(), reduce_op, rank_root, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(reduceSend);

}

template <class T, Device D>
void scheduleReduceRecvInPlace(const comm::Executor& ex,
                               pika::future<common::PromiseGuard<comm::Communicator>> pcomm,
                               MPI_Op reduce_op, pika::future<matrix::Tile<T, D>> tile) {
  // Note:
  //
  // GPU  -> duplicateIfNeeded ---------------------> cCPU -> MPI -------------> copyIfNeeded --> GPU
  // CPU  ----------------------> makeItContiguous -> cCPU -> MPI -> copyBack ------------------> CPU
  // cCPU ------------------------------------------> cCPU -> MPI ------------------------------> cCPU
  //
  // where: cCPU = contiguous CPU

  using pika::dataflow;
  using pika::unwrapping;

  using common::internal::copyBack_o;
  using common::internal::makeItContiguous_o;
  using internal::reduceRecvInPlace_o;
  using matrix::Tile;
  using matrix::duplicateIfNeeded;
  using matrix::getUnwrapReturnValue;
  using matrix::unwrapExtendTiles;

  const auto& ex_copy = getHpExecutor<Backend::MC>();

  // Note:
  //
  // TILE_ORIG can be on CPU or GPU
  //
  // TILE_CPU ----> duplicateIfNeeded<CPU> ----> TILE_CPU (no-op)
  // TILE_GPU ----> duplicateIfNeeded<CPU> ----> TILE_CPU

  pika::shared_future<Tile<T, D>> tile_orig = tile.share();
  pika::shared_future<Tile<T, Device::CPU>> tile_cpu = duplicateIfNeeded<Device::CPU>(tile_orig);

  // Note:
  //
  // TILE_CPU -+-> makeContiguous --> CONT_BUF --> mpi --> CONT_BUF -+-> copyBack --> CHECKPOINT -->
  //           |                                                     |
  //           +------------------------> TILE_CPU ------------------+----------------------------->
  //
  // CHECKPOINT assures that the copyBack has finished, which is useful for the next step.

  auto cont_buf = dataflow(ex_copy, unwrapping(makeItContiguous_o), tile_cpu);
  cont_buf =
      dataflow(ex, unwrapping(reduceRecvInPlace_o), std::move(pcomm), reduce_op, std::move(cont_buf));
  auto checkpoint = dataflow(ex_copy, unwrapExtendTiles(copyBack_o), std::move(cont_buf), tile_cpu);

  // Note:
  //
  // In case the original tile was on CPU, the previous step already managed to copy the result in place
  // and nothing else has to be done.
  // Otherwise, if it was on GPU, a temporary tile on CPU has been used, so this step handles the needed
  // copy from CPU to GPU.
  //
  // TILE_CPU => no-op
  // TILE_GPU => copy back from CPU to GPU

  copyIfNeeded(tile_cpu, tile_orig, std::move(checkpoint));
}

template <class T, Device D, template <class> class Future>
void scheduleReduceSend(const comm::Executor& ex, comm::IndexT_MPI rank_root,
                        pika::future<common::PromiseGuard<comm::Communicator>> pcomm, MPI_Op reduce_op,
                        Future<matrix::Tile<T, D>> tile) {
  using pika::dataflow;
  using pika::unwrapping;

  using common::internal::makeItContiguous_o;
  using matrix::Tile;
  using matrix::duplicateIfNeeded;
  using internal::reduceSend_o;
  using matrix::unwrapExtendTiles;

  const auto& ex_copy = getHpExecutor<Backend::MC>();

  // Note:
  //
  // TILE can be on CPU or GPU
  //
  // TILE_CPU ----> duplicateIfNeeded<CPU> ----> TILE_CPU (no-op)
  // TILE_GPU ----> duplicateIfNeeded<CPU> ----> TILE_CPU
  pika::shared_future<Tile<const T, Device::CPU>> tile_cpu =
      duplicateIfNeeded<Device::CPU>(std::move(tile));

  // Note:
  //
  // TILE_CPU -+-> makeContiguous ---> CONT_BUF ---+--> mpi_call--+
  //           |                                   |              |
  //           +-----------> TILE_CPU -------------+--------------+
  auto cont_buf = dataflow(ex_copy, unwrapping(makeItContiguous_o), tile_cpu);

  dataflow(ex, unwrapExtendTiles(reduceSend_o), rank_root, std::move(pcomm), reduce_op,
           std::move(cont_buf), tile_cpu);
}
}
}
