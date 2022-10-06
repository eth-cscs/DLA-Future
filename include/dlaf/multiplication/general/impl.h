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
                                const SizeType idx_last, const SizeType nrefls, const T alpha,
                                Matrix<const T, D>& mat_a, Matrix<const T, D>& mat_b, const T beta,
                                Matrix<T, D>& mat_c) {
  namespace ex = pika::execution::experimental;

  const auto& dist_a = mat_a.distribution();
  const auto rank = dist_a.rankIndex();

  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator().clone());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator().clone());

  const SizeType idx_end = std::min(idx_last + 1, dist_a.nrTiles().rows());
  const SizeType i_beg = dist_a.template nextLocalTileFromGlobalTile<Coord::Row>(idx_begin);
  const SizeType i_end = dist_a.template nextLocalTileFromGlobalTile<Coord::Row>(idx_end);

  const SizeType j_beg = dist_a.template nextLocalTileFromGlobalTile<Coord::Col>(idx_begin);
  const SizeType j_end = dist_a.template nextLocalTileFromGlobalTile<Coord::Col>(idx_end);

  const SizeType mb = dist_a.blockSize().rows();

  // Note:
  // Workspace needed is limited to the range [i_begin:i_last], but it can be constrained even more
  // to store just nrefls rows or cols, for column and row panel respectively.
  // The panel will be allocated just for that range, by creating an ad hoc distribution that starts
  // in the origin of the matrix an with a size covering all needed elements. Then, when the panel
  // will be created, the part before the range will be shrinked and it won't be allocated by
  // specifying the offset.
  const GlobalTileIndex panel_offset(idx_begin, idx_begin);
  const SizeType till_k = idx_begin * mb + nrefls;
  const matrix::Distribution dist_panel({till_k, till_k}, dist_a.blockSize(), dist_a.commGridSize(),
                                        dist_a.rankIndex(), dist_a.sourceRankIndex());

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, D>> panelsA(n_workspaces, dist_panel, panel_offset);
  common::RoundRobin<matrix::Panel<Coord::Row, T, D>> panelsB(n_workspaces, dist_panel, panel_offset);

  // Note:
  // This helper lambda, given an index and its end (first index after the last valid one), returns
  // two information:
  // - it returns true if not all elements of the tile are interesting (last tile + partially used)
  // - the number of interesting elements at the specified tile index, along the axis of the index
  //
  // This helps when splitting the tile to extract just relevant part involved in the computation.
  //
  // Tiles are going to be splitted because either just a subset of rows or columns are involved.
  // The first boolean information allows a minor optimization, allowing to avoid a splitTile in
  // case the full tile is going to be used.
  const bool isEndRangePartiallyUsed = ((nrefls % mb) != 0);
  const SizeType partialSize = (nrefls % mb);
  const auto sizeOfIndexInRange = [mb, &isEndRangePartiallyUsed, &partialSize](const SizeType index,
                                                                               const SizeType end) {
    const bool isLastRow = (index == (end - 1));
    const bool isPartial = (isLastRow && isEndRangePartiallyUsed);
    const SizeType size = isPartial ? partialSize : mb;
    return std::tuple{isPartial, size};
  };

  for (SizeType k = idx_begin; k <= idx_last; ++k) {
    auto& panelA = panelsA.nextResource();
    auto& panelB = panelsB.nextResource();

    const auto [isKPartial, kSize] = sizeOfIndexInRange(k, idx_last + 1);
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
        const auto [isRowPartial, nrows] = sizeOfIndexInRange(i, i_end);
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
        const auto [isColPartial, ncols] = sizeOfIndexInRange(j, j_end);
        panelB.setTile(kj, (isKPartial || isColPartial)
                               ? splitTile(mat_b.read(kj), {{0, 0}, {kSize, ncols}})
                               : mat_b.read(kj));
      }
    }

    // Broadcast both column and row panel from root to others (row-wise and col-wise, respectively)
    broadcast(rank.col(), panelA, mpi_row_task_chain);
    broadcast(rank.row(), panelB, mpi_col_task_chain);

    // This is the core loop where the k step performs the update step over the full local matrix
    // using the col and row workspaces.
    for (SizeType i = i_beg; i < i_end; ++i) {
      const auto [isRowPartial, nrows] = sizeOfIndexInRange(i, i_end);
      for (SizeType j = j_beg; j < j_end; ++j) {
        const LocalTileIndex ij(i, j);
        const auto [isColPartial, ncols] = sizeOfIndexInRange(j, j_end);

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
