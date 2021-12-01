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

  using namespace pika::execution::experimental;
  using dlaf::internal::whenAllLift;

  // this helper returns a reference to the original tile in case it is already on CPU, otherwise
  // returns a newly allocated tile whose content is copied from the one passed as parameter.
  // So, it can either return a const Tile& or a Tile, but for sure on CPU.
  constexpr auto helperTileOnCPU = [](auto& tile) constexpr->decltype(auto) {
    if constexpr (Device::CPU == D)
      return std::cref(tile);
    else
      return matrix::Duplicate<Device::CPU>{}(tile);
  };

  // the original tile has to be kept alive till the very end
  let_value(std::move(tile), [=, pcomm = std::move(pcomm)](auto& tile_orig) mutable {
    // inside next scope tile_on_cpu is for sure a tile on CPU, and it can either be the original one
    // or a new one created thanks to the helper.
    // on the other end, tile_orig is passed for the eventual copy of the result back to the original
    // device. it must be noted that it is not useful for helperTileOnCPU correctness, because its
    // functionality is ensured by the previous scope.
    return whenAllLift(std::cref(tile_orig), helperTileOnCPU(tile_orig)) |
           let_value(
               [=, pcomm = std::move(pcomm)](const matrix::Tile<T, D>& tile_orig,
                                             const matrix::Tile<T, Device::CPU>& tile_on_cpu) mutable {
                 using common::internal::makeItContiguous;

                 // inside next scope, together with the promiseguard for the communicator, we pass
                 // also the tile_on_cpu that is ensured to be contiguous thanks to makeItContiguous
                 auto comm_on_cpu =
                     whenAllLift(std::move(pcomm), makeItContiguous(tile_on_cpu),
                                 std::cref(tile_on_cpu)) |
                     let_value([=](auto& pcomm, auto& cont_buf,
                                   const matrix::Tile<T, Device::CPU>& tile_on_cpu) {
                       // communicate the tile and, just in case a temporary contiguous tile on CPU has
                       // been used, copy it back to the non-contiguous tile on CPU
                       return pika::dataflow(ex, internal::reduceRecvInPlace<T>, std::move(pcomm),
                                            reduce_op, std::cref(cont_buf)) |
                              then([&]() {
                                // TODO ref_wrapper to matrix::Tile const& is not converted
                                dlaf::common::internal::copyBack(cont_buf, tile_on_cpu);
                              });
                     });

                 // if the original tile was on CPU, nothing else has to be done. task finished
                 if constexpr (Device::CPU == D)
                   return comm_on_cpu;
                 else  // otherwise the result on CPU has to be copied back to the original device
                   // TODO Fix with PR on master about whenAll with void
                   // TODO matrix::copy(Policy<Backend::GPU>(pika::threads::thread_priority::high));
                   return whenAllLift(/*std::move(comm_on_cpu),*/ std::cref(tile_orig),
                                      std::cref(tile_on_cpu)) |
                          then([](const auto&, const auto&) {});
               });
  }) | start_detached();
}

template <class T, Device D, template <class> class Future>
void scheduleReduceSend(const comm::Executor& ex, comm::IndexT_MPI rank_root,
                        pika::future<common::PromiseGuard<comm::Communicator>> pcomm, MPI_Op reduce_op,
                        Future<matrix::Tile<T, D>> tile) {
  using namespace pika::execution::experimental;
  using dlaf::internal::whenAllLift;

  constexpr auto tileOnCPU = [](auto fut_tile) constexpr {
    if constexpr (Device::CPU == D)
      return fut_tile;
    else
      return matrix::duplicateIfNeeded<Device::CPU>(std::move(fut_tile));
  };

  // ensure that tile is on the CPU (re-use it or copy it from GPU)
  keep_future(tileOnCPU(std::move(tile))) |
      // keep it alive till the end of the communication
      let_value([ex, rank_root, reduce_op, pcomm = std::move(pcomm)](auto& fut_tile_cpu) mutable {
        // make it contiguous and keep it alive for the communication
        using common::internal::makeItContiguous;
        return whenAllLift(std::move(pcomm), makeItContiguous(fut_tile_cpu.get())) |
               let_value([ex, rank_root, reduce_op](auto& pcomm, const auto& cont_buf) {
                 return pika::dataflow(ex, internal::reduceSend<T>, rank_root, std::move(pcomm),
                                      reduce_op, std::cref(cont_buf));
               });
      }) |
      start_detached();
}
}
}
