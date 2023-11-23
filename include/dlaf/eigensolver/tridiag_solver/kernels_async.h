//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <dlaf/eigensolver/tridiag_solver/kernels.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/sender/traits.h>
#include <dlaf/sender/transform.h>
#include <dlaf/types.h>

namespace dlaf::eigensolver::internal {

template <class DiagTileSenderIn, class DiagTileSenderOut>
void stedcAsync(DiagTileSenderIn&& in, DiagTileSenderOut&& out) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;
  using pika::execution::thread_stacksize;

  auto sender = ex::when_all(std::forward<DiagTileSenderIn>(in), std::forward<DiagTileSenderOut>(out));
  ex::start_detached(tile::stedc(di::Policy<Backend::MC>(thread_stacksize::nostack), std::move(sender)));
}

template <Device D, class InTileSender, class OutTileSender>
void castToComplexAsync(InTileSender&& in, OutTileSender&& out) {
  namespace di = dlaf::internal;
  namespace ex = pika::execution::experimental;
  using pika::execution::thread_stacksize;
  auto sender = ex::when_all(std::forward<InTileSender>(in), std::forward<OutTileSender>(out));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(thread_stacksize::nostack), castToComplex_o,
                      std::move(sender));
}

template <class T, class TopTileSender, class BottomTileSender>
auto cuppensDecompAsync(TopTileSender&& top, BottomTileSender&& bottom) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;
  using pika::execution::thread_stacksize;

  constexpr auto backend = Backend::MC;

  return ex::when_all(std::forward<TopTileSender>(top), std::forward<BottomTileSender>(bottom)) |
         di::transform(di::Policy<backend>(thread_stacksize::nostack), cuppensDecomp_o);
}

template <Device D, class TridiagTile, class DiagTile>
void copyDiagonalFromCompactTridiagonalAsync(TridiagTile&& in, DiagTile&& out) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;
  using pika::execution::thread_stacksize;

  auto sender = ex::when_all(std::forward<TridiagTile>(in), std::forward<DiagTile>(out));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(thread_stacksize::nostack),
                      copyDiagonalFromCompactTridiagonal_o, std::move(sender));
}

template <class T, Device D, class RhoSender, class EvecsTileSender, class Rank1TileSender>
void assembleRank1UpdateVectorTileAsync(bool top_tile, RhoSender&& rho, EvecsTileSender&& evecs,
                                        Rank1TileSender&& rank1) {
  namespace di = dlaf::internal;
  using pika::execution::thread_stacksize;
  auto sender =
      di::whenAllLift(top_tile, std::forward<RhoSender>(rho), std::forward<EvecsTileSender>(evecs),
                      std::forward<Rank1TileSender>(rank1));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(thread_stacksize::nostack),
                      assembleRank1UpdateVectorTile_o, std::move(sender));
}
}
