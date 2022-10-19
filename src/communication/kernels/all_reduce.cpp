//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <complex>
#include <utility>

#include <mpi.h>
#include <pika/execution.hpp>

#include "dlaf/common/callable_object.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/kernels/all_reduce.h"
#include "dlaf/communication/message.h"
#include "dlaf/communication/rdma.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/transform_mpi.h"
#include "dlaf/sender/with_temporary_tile.h"

namespace dlaf::comm {
namespace internal {
template <class T, Device D>
auto allReduce(const Communicator& comm, MPI_Op reduce_op, const matrix::Tile<const T, D>& tile_in,
               const matrix::Tile<T, D>& tile_out, MPI_Request* req) {
  DLAF_ASSERT(tile_in.is_contiguous(), "");
  DLAF_ASSERT(tile_out.is_contiguous(), "");

  auto msg_in = comm::make_message(common::make_data(tile_in));
  auto msg_out = comm::make_message(common::make_data(tile_out));
  DLAF_MPI_CHECK_ERROR(MPI_Iallreduce(msg_in.data(), msg_out.data(), msg_in.count(), msg_in.mpi_type(),
                                      reduce_op, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(allReduce);

template <class TileInSender, class TileOutSender>
[[nodiscard]] auto scheduleAllReduce(
    pika::execution::experimental::unique_any_sender<dlaf::common::PromiseGuard<Communicator>> pcomm,
    MPI_Op reduce_op, TileInSender&& tile_in, TileOutSender&& tile_out) {
  using dlaf::comm::CommunicationDevice_v;
  using dlaf::comm::internal::allReduce_o;
  using dlaf::comm::internal::transformMPI;
  using dlaf::internal::CopyFromDestination;
  using dlaf::internal::CopyToDestination;
  using dlaf::internal::RequireContiguous;
  using dlaf::internal::whenAllLift;
  using dlaf::internal::withTemporaryTile;

  // We create two nested scopes for the input and output tiles with
  // withTemporaryTile. The output tile is in the outer scope as the output tile
  // will be returned by the returned sender.
  auto all_reduce_final = [reduce_op, pcomm = std::move(pcomm),
                           tile_in =
                               std::forward<TileInSender>(tile_in)](auto const& tile_out_comm) mutable {
    auto all_reduce = [reduce_op, pcomm = std::move(pcomm),
                       &tile_out_comm](auto const& tile_in_comm) mutable {
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
    constexpr static Device in_device = SenderSingleValueType<TileInSender>::device;
    constexpr static Device in_comm_device = CommunicationDevice_v<in_device>;

    return withTemporaryTile<in_comm_device, CopyToDestination::Yes, CopyFromDestination::No,
                             RequireContiguous::Yes>(std::move(tile_in), std::move(all_reduce));
  };

  // The output tile does not need to be copied to the temporary tile since it
  // is only written to. The written data is copied back from the temporary tile
  // to the output tile. The reduction is explicitly done on CPU memory so that
  // we can manage potential asynchronous copies between CPU and GPU. A
  // reduction requires contiguous memory.
  constexpr static Device out_device = SenderSingleValueType<TileOutSender>::device;
  constexpr static Device out_comm_device = CommunicationDevice_v<out_device>;

  return withTemporaryTile<out_comm_device, CopyToDestination::No, CopyFromDestination::Yes,
                           RequireContiguous::Yes>(std::forward<TileOutSender>(tile_out),
                                                   std::move(all_reduce_final));
}

template <class T, Device D>
auto allReduceInPlace(const Communicator& comm, MPI_Op reduce_op, const matrix::Tile<T, D>& tile,
                      MPI_Request* req) {
  DLAF_ASSERT(tile.is_contiguous(), "");

  auto msg = comm::make_message(common::make_data(tile));
  DLAF_MPI_CHECK_ERROR(
      MPI_Iallreduce(MPI_IN_PLACE, msg.data(), msg.count(), msg.mpi_type(), reduce_op, comm, req));
}

DLAF_MAKE_CALLABLE_OBJECT(allReduceInPlace);
}

template <class T, Device D>
[[nodiscard]] pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> scheduleAllReduce(
    pika::execution::experimental::unique_any_sender<dlaf::common::PromiseGuard<Communicator>> pcomm,
    MPI_Op reduce_op, pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> tile_in,
    pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> tile_out) {
  return internal::scheduleAllReduce(std::move(pcomm), reduce_op, std::move(tile_in),
                                     std::move(tile_out));
}

DLAF_SCHEDULE_ALL_REDUCE_ETI(, float, Device::CPU);
DLAF_SCHEDULE_ALL_REDUCE_ETI(, double, Device::CPU);
DLAF_SCHEDULE_ALL_REDUCE_ETI(, std::complex<float>, Device::CPU);
DLAF_SCHEDULE_ALL_REDUCE_ETI(, std::complex<double>, Device::CPU);

template <class T, Device D>
[[nodiscard]] pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> scheduleAllReduce(
    pika::execution::experimental::unique_any_sender<dlaf::common::PromiseGuard<Communicator>> pcomm,
    MPI_Op reduce_op,
    pika::execution::experimental::unique_any_sender<pika::shared_future<matrix::Tile<const T, D>>>
        tile_in,
    pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> tile_out) {
  return internal::scheduleAllReduce(std::move(pcomm), reduce_op, std::move(tile_in),
                                     std::move(tile_out));
}

DLAF_SCHEDULE_ALL_REDUCE_SFTILE_ETI(, int, Device::CPU);
DLAF_SCHEDULE_ALL_REDUCE_SFTILE_ETI(, float, Device::CPU);
DLAF_SCHEDULE_ALL_REDUCE_SFTILE_ETI(, double, Device::CPU);
DLAF_SCHEDULE_ALL_REDUCE_SFTILE_ETI(, std::complex<float>, Device::CPU);
DLAF_SCHEDULE_ALL_REDUCE_SFTILE_ETI(, std::complex<double>, Device::CPU);

template <class T, Device D>
[[nodiscard]] pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> scheduleAllReduceInPlace(
    pika::execution::experimental::unique_any_sender<dlaf::common::PromiseGuard<Communicator>> pcomm,
    MPI_Op reduce_op, pika::execution::experimental::unique_any_sender<matrix::Tile<T, D>> tile) {
  using dlaf::comm::CommunicationDevice_v;
  using dlaf::comm::internal::allReduceInPlace_o;
  using dlaf::comm::internal::transformMPI;
  using dlaf::internal::CopyFromDestination;
  using dlaf::internal::CopyToDestination;
  using dlaf::internal::RequireContiguous;
  using dlaf::internal::whenAllLift;
  using dlaf::internal::withTemporaryTile;

  constexpr static auto D = dlaf::internal::SenderSingleValueType<TileSender>::device;

  auto all_reduce_in_place = [reduce_op, pcomm = std::move(pcomm)](auto const& tile_comm) mutable {
    return whenAllLift(std::move(pcomm), reduce_op, std::cref(tile_comm)) |
           transformMPI(allReduceInPlace_o);
  };

  // The tile has to be copied both to and from the temporary tile since the
  // reduction is done in-place. The reduction is explicitly done on CPU memory
  // so that we can manage potential asynchronous copies between CPU and GPU. A
  // reduction requires contiguous memory.
  return withTemporaryTile<CommunicationDevice_v<D>, CopyToDestination::Yes, CopyFromDestination::Yes,
                           RequireContiguous::Yes>(std::move(tile), std::move(all_reduce_in_place));
}

DLAF_SCHEDULE_ALL_REDUCE_IN_PLACE_ETI(, int, Device::CPU);
DLAF_SCHEDULE_ALL_REDUCE_IN_PLACE_ETI(, float, Device::CPU);
DLAF_SCHEDULE_ALL_REDUCE_IN_PLACE_ETI(, double, Device::CPU);
DLAF_SCHEDULE_ALL_REDUCE_IN_PLACE_ETI(, std::complex<float>, Device::CPU);
DLAF_SCHEDULE_ALL_REDUCE_IN_PLACE_ETI(, std::complex<double>, Device::CPU);
}
