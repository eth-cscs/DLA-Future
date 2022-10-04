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

#include "dlaf/multiplication/general/api.h"

#include "dlaf/blas/tile.h"
#include "dlaf/common/assert.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/communication/broadcast_panel.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/sender/when_all_lift.h"

namespace dlaf::multiplication {
namespace internal {

template <Backend B, Device D, class T>
void GeneralSub<B, D, T>::callNN(const SizeType idx_begin, const SizeType idx_end, const blas::Op opA,
                                 const blas::Op opB, const T alpha, Matrix<const T, D>& mat_a,
                                 Matrix<const T, D>& mat_b, const T beta, Matrix<T, D>& mat_c) {
  namespace ex = pika::execution::experimental;

  for (SizeType j = idx_begin; j <= idx_end; ++j) {
    for (SizeType i = idx_begin; i <= idx_end; ++i) {
      for (SizeType k = idx_begin; k <= idx_end; ++k) {
        ex::start_detached(dlaf::internal::whenAllLift(opA, opB, alpha,
                                                       mat_a.read_sender(GlobalTileIndex(i, k)),
                                                       mat_b.read_sender(GlobalTileIndex(k, j)),
                                                       k == idx_begin ? beta : T(1),
                                                       mat_c.readwrite_sender(GlobalTileIndex(i, j))) |
                           tile::gemm(dlaf::internal::Policy<B>()));
      }
    }
  }
}

template <Backend B, Device D, class T>
void GeneralSubK<B, D, T>::call(comm::CommunicatorGrid grid, const SizeType idx_begin,
                                const SizeType idx_last, const SizeType k, const T alpha,
                                Matrix<const T, D>& mat_a, Matrix<const T, D>& mat_b, const T beta,
                                Matrix<T, D>& mat_c) {
  namespace ex = pika::execution::experimental;

  const auto& dist_a = mat_a.distribution();
  const auto& dist_c = mat_c.distribution();
  const auto rank = dist_c.rankIndex();

  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator().clone());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator().clone());

  // TODO minimize panel memory allocation (in order to reduce also communications)
  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, D>> col_panels(n_workspaces, dist_c);
  common::RoundRobin<matrix::Panel<Coord::Row, T, D>> row_panels(n_workspaces, dist_c);

  const SizeType idx_end = std::min(idx_last + 1, dist_a.nrTiles().rows());
  const SizeType i_beg = dist_a.template nextLocalTileFromGlobalTile<Coord::Row>(idx_begin);
  const SizeType i_end = dist_a.template nextLocalTileFromGlobalTile<Coord::Row>(idx_end);

  const SizeType j_beg = dist_a.template nextLocalTileFromGlobalTile<Coord::Col>(idx_begin);
  const SizeType j_end = dist_a.template nextLocalTileFromGlobalTile<Coord::Col>(idx_end);

  for (SizeType k = idx_begin; k <= idx_last; ++k) {
    auto& col_panel = col_panels.nextResource();
    auto& row_panel = row_panels.nextResource();

    const auto rank_k = dist_a.template rankGlobalTile<Coord::Col>(k);
    if (rank_k == rank.col()) {
      const auto k_local = dist_a.template localTileFromGlobalTile<Coord::Col>(k);
      for (SizeType i = i_beg; i < i_end; ++i) {
        const LocalTileIndex i0(i, 0);
        col_panel.setTile(i0, mat_a.read(LocalTileIndex{i, k_local}));
      }
    }
    if (rank_k == rank.row()) {
      const auto k_local = dist_a.template localTileFromGlobalTile<Coord::Row>(k);
      for (SizeType j = j_beg; j < j_end; ++j) {
        const LocalTileIndex j0(0, j);
        row_panel.setTile(j0, mat_b.read(LocalTileIndex{k_local, j}));
      }
    }

    broadcast(rank.col(), col_panel, mpi_row_task_chain);
    broadcast(rank.row(), row_panel, mpi_col_task_chain);

    for (SizeType i = i_beg; i < i_end; ++i) {
      for (SizeType j = j_beg; j < j_end; ++j) {
        // TODO split last tile if not all elements are used

        ex::start_detached(dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, alpha,
                                                       col_panel.read_sender(LocalTileIndex{i, 0}),
                                                       row_panel.read_sender(LocalTileIndex{0, j}),
                                                       k == idx_begin ? beta : T(1),
                                                       mat_c.readwrite_sender(LocalTileIndex(i, j))) |
                           tile::gemm(dlaf::internal::Policy<B>()));
      }
    }

    col_panel.reset();
    row_panel.reset();
  }
}
}
}
