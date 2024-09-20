//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <complex>
#include <functional>
#include <utility>

#include <mpi.h>

#include <pika/execution.hpp>
#include <pika/execution_base/any_sender.hpp>

#include <dlaf/common/callable_object.h>
#include <dlaf/common/eti.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/communication/kernels/all_reduce.h>
#include <dlaf/communication/message.h>
#include <dlaf/communication/rdma.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/sender/transform_mpi.h>
#include <dlaf/sender/with_temporary_tile.h>

namespace dlaf::comm::internal {
template <class T, Device DIn, Device DOut>
auto allReduce(const Communicator& comm, MPI_Op reduce_op, const matrix::Tile<const T, DIn>& tile_in,
               const matrix::Tile<T, DOut>& tile_out, MPI_Request* req) {
#if !defined(DLAF_WITH_MPI_GPU_AWARE)
  static_assert(DIn == Device::CPU, "DLAF_WITH_MPI_GPU_AWARE=off, MPI accepts only CPU memory.");
  static_assert(DOut == Device::CPU, "DLAF_WITH_MPI_GPU_AWARE=off, MPI accepts only CPU memory.");
#endif
  DLAF_ASSERT(tile_in.is_contiguous(), "");
  DLAF_ASSERT(tile_out.is_contiguous(), "");

  auto msg_in = comm::make_message(common::make_data(tile_in));
  auto msg_out = comm::make_message(common::make_data(tile_out));
  DLAF_MPI_CHECK_ERROR(MPI_Iallreduce(msg_in.data(), msg_out.data(), msg_in.count(), msg_in.mpi_type(),
                                      reduce_op, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(allReduce);

template <Device DCommIn, Device DCommOut, class T, Device DIn, Device DOut>
[[nodiscard]] auto scheduleAllReduce(
    pika::execution::experimental::unique_any_sender<CommunicatorPipelineExclusiveWrapper> pcomm,
    MPI_Op reduce_op, dlaf::matrix::ReadOnlyTileSender<T, DIn> tile_in,
    dlaf::matrix::ReadWriteTileSender<T, DOut> tile_out) {
  using dlaf::comm::CommunicationDevice_v;
  using dlaf::comm::internal::allReduce_o;
  using dlaf::comm::internal::transformMPI;
  using dlaf::internal::CopyFromDestination;
  using dlaf::internal::CopyToDestination;
  using dlaf::internal::RequireContiguous;
  using dlaf::internal::SenderSingleValueType;
  using dlaf::internal::whenAllLift;
  using dlaf::internal::withTemporaryTile;

  // We create two nested scopes for the input and output tiles with
  // withTemporaryTile. The output tile is in the outer scope as the output tile
  // will be returned by the returned sender.
  auto all_reduce_final = [reduce_op, pcomm = std::move(pcomm),
                           tile_in = std::move(tile_in)](const auto& tile_out_comm) mutable {
    auto all_reduce = [reduce_op, pcomm = std::move(pcomm),
                       &tile_out_comm](const auto& tile_in_comm) mutable {
      return whenAllLift(std::move(pcomm), reduce_op, std::cref(tile_in_comm),
                         std::cref(tile_out_comm)) |
             transformMPI(allReduce_o);
    };
    // The input tile must be copied to the temporary tile used for the
    // reduction, but the temporary tile does not need to be copied back to the
    // input since the data is not changed by the reduction (the result is
    // written into the output tile instead).  The reduction is explicitly done
    // on CPU memory so that we can manage potential asynchronous copies between
    // CPU and GPU. A reduction requires contiguous memory.
    return pika::execution::experimental::make_unique_any_sender(
        withTemporaryTile<DCommIn, CopyToDestination::Yes, CopyFromDestination::No,
                          RequireContiguous::Yes>(std::move(tile_in), std::move(all_reduce)));
  };

#if !defined(DLAF_WITH_MPI_GPU_AWARE)
  static_assert(DCommIn == Device::CPU, "DLAF_WITH_MPI_GPU_AWARE=off, MPI accepts only CPU memory.");
  static_assert(DCommOut == Device::CPU, "DLAF_WITH_MPI_GPU_AWARE=off, MPI accepts only CPU memory.");
#endif

  // The output tile does not need to be copied to the temporary tile since it
  // is only written to. The written data is copied back from the temporary tile
  // to the output tile. The reduction is explicitly done on CPU memory so that
  // we can manage potential asynchronous copies between CPU and GPU. A
  // reduction requires contiguous memory.
  return withTemporaryTile<DCommOut, CopyToDestination::No, CopyFromDestination::Yes,
                           RequireContiguous::Yes>(std::move(tile_out), std::move(all_reduce_final));
}

template <class T, Device D>
auto allReduceInPlace(const Communicator& comm, MPI_Op reduce_op, const matrix::Tile<T, D>& tile,
                      MPI_Request* req) {
#if !defined(DLAF_WITH_MPI_GPU_AWARE)
  static_assert(D == Device::CPU, "DLAF_WITH_MPI_GPU_AWARE=off, MPI accepts only CPU memory.");
#endif
  DLAF_ASSERT(tile.is_contiguous(), "");

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(MPI_Iallreduce(MPI_IN_PLACE, msg.data(), msg.count(), msg.mpi_type(), reduce_op,
                                      comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(allReduceInPlace);

template <Device DComm, class T, Device D>
[[nodiscard]] dlaf::matrix::ReadWriteTileSender<T, D> schedule_all_reduce_in_place(
    pika::execution::experimental::unique_any_sender<CommunicatorPipelineExclusiveWrapper> pcomm,
    MPI_Op reduce_op, dlaf::matrix::ReadWriteTileSender<T, D> tile) {
  using dlaf::comm::CommunicationDevice_v;
  using dlaf::comm::internal::allReduceInPlace_o;
  using dlaf::comm::internal::transformMPI;
  using dlaf::internal::CopyFromDestination;
  using dlaf::internal::CopyToDestination;
  using dlaf::internal::RequireContiguous;
  using dlaf::internal::whenAllLift;
  using dlaf::internal::withTemporaryTile;

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
  auto all_reduce_in_place = [reduce_op, pcomm = std::move(pcomm)](const auto& tile_comm) mutable {
    return whenAllLift(std::move(pcomm), reduce_op, std::cref(tile_comm)) |
           transformMPI(allReduceInPlace_o);
  };
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#if !defined(DLAF_WITH_MPI_GPU_AWARE)
  static_assert(DComm == Device::CPU, "DLAF_WITH_MPI_GPU_AWARE=off, MPI accepts only CPU memory.");
#endif

  // The tile has to be copied both to and from the temporary tile since the
  // reduction is done in-place. The reduction is explicitly done on CPU memory
  // so that we can manage potential asynchronous copies between CPU and GPU. A
  // reduction requires contiguous memory.
  return withTemporaryTile<DComm, CopyToDestination::Yes, CopyFromDestination::Yes,
                           RequireContiguous::Yes>(std::move(tile), std::move(all_reduce_in_place));
}

}
