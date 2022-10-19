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

/// @file

#include <type_traits>
#include <utility>

#include <mpi.h>
#include <pika/execution.hpp>

#include "dlaf/blas/tile_extensions.h"
#include "dlaf/common/assert.h"
#include "dlaf/common/callable_object.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/kernels/p2p.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/traits.h"
#include "dlaf/sender/when_all_lift.h"

namespace dlaf::comm {

/// Schedule a P2P AllReduce Sum operation between current rank and `rank_mate`.
///
/// A P2P AllReduce Sum operation, i.e. an all-reduce involving just two ranks with MPI_SUM as op,
/// is performed between the rank where this function is called and `rank_mate`.
///
/// `in` is a sender of a read-only tile that, together with the one received from `rank_mate`, will be
/// summed in `out` (on both ranks)
template <Backend B, class CommSender, class SenderIn, class SenderOut>
[[nodiscard]] auto scheduleAllSumP2P(CommSender&& comm, IndexT_MPI rank_mate, IndexT_MPI tag,
                                     SenderIn&& in, SenderOut&& out) {
  namespace ex = pika::execution::experimental;

  using T = dlaf::internal::SenderElementType<SenderIn>;

  static_assert(std::is_same_v<T, dlaf::internal::SenderElementType<SenderOut>>,
                "in and out should send a tile of the same type");

  // Note:
  // Each rank in order to locally complete the operation just need to receive the other rank
  // data and then do the reduce operation. For this reason, the send operation is scheduled
  // independently from the rest of the allreduce operation.
  ex::start_detached(comm::scheduleSend(ex::make_unique_any_sender(comm), rank_mate, tag,
                                        ex::make_unique_any_sender(in)));

  auto tile_out =
      comm::scheduleRecv(ex::make_unique_any_sender(std::forward<CommSender>(comm)), rank_mate, tag,
                         pika::execution::experimental::make_unique_any_sender(
                             std::forward<SenderOut>(out)));
  return dlaf::internal::whenAllLift(T(1), std::forward<SenderIn>(in), std::move(tile_out)) |
         tile::add(dlaf::internal::Policy<B>());
}

}
