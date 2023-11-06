//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#ifdef DLAF_WITH_CUDA_RDMA
#warning "Reduce is not using CUDA_RDMA."
#endif

#include <complex>
#include <utility>

#include <mpi.h>

#include <pika/execution.hpp>

#include <dlaf/common/data.h>
#include <dlaf/common/eti.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/kernels/reduce.h>
#include <dlaf/communication/message.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/sender/traits.h>
#include <dlaf/sender/transform_mpi.h>
#include <dlaf/sender/with_temporary_tile.h>

namespace dlaf::comm {
namespace internal {
template <class T, Device D>
void reduceRecvInPlace(const Communicator& comm, MPI_Op reduce_op, const matrix::Tile<T, D>& tile,
                       MPI_Request* req) {
  static_assert(D == Device::CPU, "reduceRecvInPlace requires CPU memory");
  DLAF_ASSERT(tile.is_contiguous(), "");

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(MPI_Ireduce(MPI_IN_PLACE, msg.data(), msg.count(), msg.mpi_type(), reduce_op,
                                   comm.rank(), comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(reduceRecvInPlace);

template <class T, Device D>
void reduceSend(const Communicator& comm, comm::IndexT_MPI rank_root, MPI_Op reduce_op,
                const matrix::Tile<const T, D>& tile, MPI_Request* req) {
  static_assert(D == Device::CPU, "reduceSend requires CPU memory");
  DLAF_ASSERT(tile.is_contiguous(), "");

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(MPI_Ireduce(msg.data(), nullptr, msg.count(), msg.mpi_type(), reduce_op,
                                   rank_root, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(reduceSend);

template <class CommSender, class TileSender>
[[nodiscard]] auto scheduleReduceSend(CommSender&& pcomm, comm::IndexT_MPI rank_root, MPI_Op reduce_op,
                                      TileSender&& tile) {
  using dlaf::comm::internal::reduceSend_o;
  using dlaf::comm::internal::transformMPI;
  using dlaf::internal::CopyFromDestination;
  using dlaf::internal::CopyToDestination;
  using dlaf::internal::RequireContiguous;
  using dlaf::internal::whenAllLift;
  using dlaf::internal::withTemporaryTile;

  auto reduce_send = [rank_root, reduce_op,
                      pcomm = std::forward<CommSender>(pcomm)](const auto& tile_comm) mutable {
    return whenAllLift(std::move(pcomm), rank_root, reduce_op, std::cref(tile_comm)) |
           transformMPI(reduceSend_o);
  };

  // The input tile must be copied to the temporary tile used for the send, but
  // the temporary tile does not need to be copied back to the input since the
  // data is not changed by the send. The reduction is explicitly done on CPU
  // memory so that we can manage potential asynchronous copies between CPU and
  // GPU. A reduction requires contiguous memory.
  return withTemporaryTile<Device::CPU, CopyToDestination::Yes, CopyFromDestination::No,
                           RequireContiguous::Yes>(std::forward<TileSender>(tile),
                                                   std::move(reduce_send));
}
}

template <class T, Device D>
[[nodiscard]] dlaf::matrix::ReadWriteTileSender<T, D> scheduleReduceRecvInPlace(
    pika::execution::experimental::unique_any_sender<CommunicatorPipeline::ReadWriteWrapper>
        pcomm,
    MPI_Op reduce_op, dlaf::matrix::ReadWriteTileSender<T, D> tile) {
  using dlaf::comm::internal::reduceRecvInPlace_o;
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
  auto reduce_recv_in_place = [reduce_op, pcomm = std::move(pcomm)](const auto& tile_comm) mutable {
    return whenAllLift(std::move(pcomm), reduce_op, std::cref(tile_comm)) |
           transformMPI(reduceRecvInPlace_o);
  };
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

  // The input tile must be copied to the temporary tile to participate in the
  // reduction. The temporary tile is also copied back so that the reduced
  // result can be used. The reduction is explicitly done on CPU memory so that
  // we can manage potential asynchronous copies between CPU and GPU. A
  // reduction requires contiguous memory.
  return withTemporaryTile<Device::CPU, CopyToDestination::Yes, CopyFromDestination::Yes,
                           RequireContiguous::Yes>(std::move(tile), std::move(reduce_recv_in_place));
}

DLAF_EXPAND_ETI_SDCZ_DEVICE(DLAF_SCHEDULE_REDUCE_RECV_IN_PLACE_ETI, );
DLAF_SCHEDULE_REDUCE_RECV_IN_PLACE_ETI(, int, Device::CPU);

template <class T, Device D>
[[nodiscard]] pika::execution::experimental::unique_any_sender<> scheduleReduceSend(
    pika::execution::experimental::unique_any_sender<CommunicatorPipeline::ReadWriteWrapper>
        pcomm,
    comm::IndexT_MPI rank_root, MPI_Op reduce_op, dlaf::matrix::ReadOnlyTileSender<T, D> tile) {
  return internal::scheduleReduceSend(std::move(pcomm), rank_root, reduce_op, std::move(tile));
}

DLAF_EXPAND_ETI_SDCZ_DEVICE(DLAF_SCHEDULE_REDUCE_SEND_ETI, );
DLAF_SCHEDULE_REDUCE_SEND_ETI(, int, Device::CPU);
}
