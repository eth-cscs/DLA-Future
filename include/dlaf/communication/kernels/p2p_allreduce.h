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

#include <functional>

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

template <class SenderIn, class SenderOut>
[[nodiscard]] auto scheduleAllReduceP2P(MPI_Op op, Communicator comm, IndexT_MPI rank_mate,
                                        IndexT_MPI tag, SenderIn&& in, SenderOut&& out) {
  DLAF_ASSERT(op == MPI_SUM, op, "MPI_SUM is the only reduce operation supported.");

  using T = dlaf::internal::SenderElementType<SenderIn>;
  static_assert(std::is_same_v<T, dlaf::internal::SenderElementType<SenderOut>>,
                "in and out should send a tile of the same type");

  namespace ex = pika::execution::experimental;

  ex::start_detached(comm::scheduleSend(comm, rank_mate, tag, in));

  auto tile_out = comm::scheduleRecv(comm, rank_mate, tag, std::forward<SenderOut>(out));
  return dlaf::internal::whenAllLift(T(1), in, std::move(tile_out)) |
         tile::add(dlaf::internal::Policy<Backend::MC>());
}

}
