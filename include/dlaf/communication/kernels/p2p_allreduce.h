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

namespace dlaf::comm {

template <class Sender>
[[nodiscard]] auto scheduleAllReduceP2P(MPI_Op op, Communicator comm, IndexT_MPI rank_mate,
                                        IndexT_MPI tag, Sender&& tile, Sender&& tile_aux) {
  // TODO is it a good API? rank_mate does not constrain involved tags. burden to the user
  // TODO is it a good API? aux buffer passed from external, or better relying on umpire and allocate temp?
  // TODO is it a good API? differentiate sender types?

  namespace ex = pika::execution::experimental;

  using dlaf::internal::whenAllLift;

  using T = dlaf::internal::SenderElementType<Sender>;

  DLAF_ASSERT(op == MPI_SUM, op, "MPI_SUM is the only reduce operation supported.");

  return dlaf::internal::whenAllLift(std::forward<Sender>(tile), std::forward<Sender>(tile_aux)) |
         ex::let_value([comm, tag, rank_mate](const matrix::Tile<T, Device::CPU>& tile,
                                              const matrix::Tile<T, Device::CPU>& tile_aux) {
           return whenAllLift(comm::scheduleSend(comm, rank_mate, tag, ex::just(std::cref(tile))) |
                                  ex::drop_value(),
                              comm::scheduleRecv(comm, rank_mate, tag, ex::just(std::cref(tile_aux))) |
                                  ex::drop_value(),
                              T(1), std::cref(tile_aux), std::cref(tile)) |
                  tile::add(dlaf::internal::Policy<Backend::MC>());
         });
}
}
