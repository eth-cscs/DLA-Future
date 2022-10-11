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

// This implementation is based on
//
// Van De Geijn, Robert A., and Jerrell Watts.
// SUMMA: Scalable universal matrix multiplication algorithm.
// Concurrency: Practice and Experience 9.4 (1997): 255-274
template <Backend B, Device D, class T>
void GeneralSubK<B, D, T>::callNN(comm::CommunicatorGrid grid, const SizeType idx_begin,
                                  const SizeType idx_last, const T alpha, Matrix<const T, D>& mat_a,
                                  Matrix<const T, D>& mat_b, const T beta, Matrix<T, D>& mat_c) {
  namespace ex = pika::execution::experimental;

  const auto& dist_a = mat_a.distribution();
  const auto rank = dist_a.rankIndex();

  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator().clone());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator().clone());

  // which rank has the last tile involved
  const bool rankHasLastRow = rank.row() == dist_a.template rankGlobalTile<Coord::Row>(idx_last);
  const bool rankHasLastCol = rank.col() == dist_a.template rankGlobalTile<Coord::Col>(idx_last);

  // translate from global to local indices
  const SizeType idx_end = idx_last + 1;
  const SizeType i_beg = dist_a.template nextLocalTileFromGlobalTile<Coord::Row>(idx_begin);
  const SizeType i_end = dist_a.template nextLocalTileFromGlobalTile<Coord::Row>(idx_end);

  const SizeType j_beg = dist_a.template nextLocalTileFromGlobalTile<Coord::Col>(idx_begin);
  const SizeType j_end = dist_a.template nextLocalTileFromGlobalTile<Coord::Col>(idx_end);

  const SizeType mb = dist_a.blockSize().rows();
  const SizeType lastTileElement = std::min(idx_end * mb - 1, dist_a.size().rows() - 1);
  const SizeType nrefls = lastTileElement - idx_begin * mb + 1;

  // Note: if last tile is incomplete, compute the size of it
  const bool isEndRangePartial = nrefls % mb != 0;
  const SizeType partialSize = (nrefls % mb);

  // Note:
  // Workspace needed is limited to the range [i_begin:i_last]. Its allocation is obtained by creating an
  // ad hoc distribution that starts in the origin of the matrix and with a size covering all needed
  // elements. This would lead to a [0:i_last] range, but by using panel offset at initialization, the
  // part before the range will be left out from allocation, actually getting [i_begin:i_last]
  const GlobalTileIndex panel_offset(idx_begin, idx_begin);
  const matrix::Distribution dist_panel({lastTileElement + 1, lastTileElement + 1}, dist_a.blockSize(),
                                        dist_a.commGridSize(), dist_a.rankIndex(),
                                        dist_a.sourceRankIndex());

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, D>> panelsA(n_workspaces, dist_panel, panel_offset);
  common::RoundRobin<matrix::Panel<Coord::Row, T, D>> panelsB(n_workspaces, dist_panel, panel_offset);

  // This loops over the global indices for k, because every rank have to participate in communication
  for (SizeType k = idx_begin; k <= idx_last; ++k) {
    auto& panelA = panelsA.nextResource();
    auto& panelB = panelsB.nextResource();

    const bool isKPartial = k == idx_last && isEndRangePartial;
    const SizeType kSize = isKPartial ? partialSize : mb;
    if (isKPartial) {
      panelA.setWidth(kSize);
      panelB.setHeight(kSize);
    }

    const auto rank_k = dist_a.rankGlobalTile({k, k});

    // Setup the column workspace for the root ranks, i.e. the ones in the current col
    if (rank_k.col() == rank.col()) {
      const auto k_local = dist_a.template localTileFromGlobalTile<Coord::Col>(k);
      for (SizeType i = i_beg; i < i_end; ++i) {
        const LocalTileIndex ik(i, k_local);
        const bool isRowPartial = (i == i_end - 1 && isEndRangePartial && rankHasLastRow);
        const SizeType nrows = isRowPartial ? partialSize : mb;
        panelA.setTile(ik, (isRowPartial || isKPartial)
                               ? splitTile(mat_a.read(ik), {{0, 0}, {nrows, kSize}})
                               : mat_a.read(ik));
      }
    }
    // Setup the row workspace for the root ranks, i.e. the ones in the current row
    if (rank_k.row() == rank.row()) {
      const auto k_local = dist_a.template localTileFromGlobalTile<Coord::Row>(k);
      for (SizeType j = j_beg; j < j_end; ++j) {
        const LocalTileIndex kj(k_local, j);
        const bool isColPartial = (j == j_end - 1 && isEndRangePartial && rankHasLastCol);
        const SizeType ncols = isColPartial ? partialSize : mb;
        panelB.setTile(kj, (isKPartial || isColPartial)
                               ? splitTile(mat_b.read(kj), {{0, 0}, {kSize, ncols}})
                               : mat_b.read(kj));
      }
    }

    // Broadcast both column and row panel from root to others (row-wise and col-wise, respectively)
    broadcast(rank_k.col(), panelA, mpi_row_task_chain);
    broadcast(rank_k.row(), panelB, mpi_col_task_chain);

    // This is the core loop where the k step performs the update over the entire local matrix using
    // the col and row workspaces.
    // Everything needed for the update is available locally thanks to previous broadcasts.
    for (SizeType i = i_beg; i < i_end; ++i) {
      const bool isRowPartial = (i == i_end - 1 && isEndRangePartial && rankHasLastRow);
      const SizeType nrows = isRowPartial ? partialSize : mb;

      for (SizeType j = j_beg; j < j_end; ++j) {
        const LocalTileIndex ij(i, j);

        const bool isColPartial = (j == j_end - 1 && isEndRangePartial && rankHasLastCol);
        const SizeType ncols = isColPartial ? partialSize : mb;

        ex::start_detached(
            dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, alpha,
                                        panelA.read_sender(ij), panelB.read_sender(ij),
                                        k == idx_begin ? beta : T(1),
                                        (isRowPartial || isColPartial)
                                            ? splitTile(mat_c(ij), {{0, 0}, {nrows, ncols}})
                                            : mat_c.readwrite_sender(ij)) |
            tile::gemm(dlaf::internal::Policy<B>()));
      }
    }

    panelA.reset();
    panelB.reset();
  }
}
}
}
