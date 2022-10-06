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
  const auto& dist_c = mat_c.distribution();
  const auto rank = dist_c.rankIndex();

  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator().clone());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator().clone());

  const SizeType idx_end = std::min(idx_last + 1, dist_a.nrTiles().rows());
  const SizeType i_beg = dist_a.template nextLocalTileFromGlobalTile<Coord::Row>(idx_begin);
  const SizeType i_end = dist_a.template nextLocalTileFromGlobalTile<Coord::Row>(idx_end);

  const SizeType j_beg = dist_a.template nextLocalTileFromGlobalTile<Coord::Col>(idx_begin);
  const SizeType j_end = dist_a.template nextLocalTileFromGlobalTile<Coord::Col>(idx_end);

  const SizeType mb = dist_a.blockSize().rows();
  const bool isKPartialTile = nrefls % mb != 0;

  // Note: minimize panel memory allocation
  // TODO In particular, it will be reduced on both head and tail
  const SizeType till_k = idx_begin * mb + nrefls;
  const matrix::Distribution dist_panel({till_k, till_k}, dist_a.blockSize(), dist_a.commGridSize(),
                                        dist_a.rankIndex(), dist_a.sourceRankIndex());

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, D>> panelsA(n_workspaces, dist_panel,
                                                              GlobalTileIndex{idx_begin, idx_begin});
  common::RoundRobin<matrix::Panel<Coord::Row, T, D>> panelsB(n_workspaces, dist_panel,
                                                              GlobalTileIndex{idx_begin, idx_begin});
  const auto iRows = [i_end, nrefls, isKPartialTile, mb](const SizeType i_loc_tile) {
    const bool isLastRow = i_loc_tile == i_end - 1;
    // TODO workaround for tile_c
    const SizeType nrows = (isLastRow && isKPartialTile) ? (nrefls % mb) : mb;
    return std::tuple{isLastRow, nrows};
  };

  const auto jCols = [j_end, nrefls, isKPartialTile, mb](const SizeType j_loc_tile) {
    const bool isLastCol = j_loc_tile == j_end - 1;
    // TODO workaround for tile_c
    const SizeType ncols = (isLastCol && isKPartialTile) ? (nrefls % mb) : mb;
    return std::tuple{isLastCol, ncols};
  };

  for (SizeType k = idx_begin; k <= idx_last; ++k) {
    auto& panelA = panelsA.nextResource();
    auto& panelB = panelsB.nextResource();

    const auto [nrows, ncols] = [&]() {
      if (k == idx_last && isKPartialTile) {
        panelA.setWidth(nrefls % mb);
        panelB.setHeight(nrefls % mb);
        return std::tuple{nrefls % mb, nrefls % mb};
      }
      return std::tuple{mb, mb};
    }();

    const auto rank_k = dist_a.rankGlobalTile({k, k});
    if (rank_k.col() == rank.col()) {
      const auto k_local = dist_a.template localTileFromGlobalTile<Coord::Col>(k);
      for (SizeType i = i_beg; i < i_end; ++i) {
        const LocalTileIndex ik(i, k_local);
        panelA.setTile(ik, splitTile(mat_a.read(ik), {{0, 0}, {std::get<1>(iRows(i)), ncols}}));
      }
    }
    if (rank_k.row() == rank.row()) {
      const auto k_local = dist_a.template localTileFromGlobalTile<Coord::Row>(k);
      for (SizeType j = j_beg; j < j_end; ++j) {
        const LocalTileIndex kj(k_local, j);
        panelB.setTile(kj, splitTile(mat_b.read(kj), {{0, 0}, {nrows, std::get<1>(jCols(j))}}));
      }
    }

    broadcast(rank.col(), panelA, mpi_row_task_chain);
    broadcast(rank.row(), panelB, mpi_col_task_chain);

    for (SizeType i = i_beg; i < i_end; ++i) {
      const auto [isLastRow, nrows] = iRows(i);
      for (SizeType j = j_beg; j < j_end; ++j) {
        const auto [isLastCol, ncols] = jCols(j);

        const LocalTileIndex ij(i, j);

        auto tile_c = (!isLastRow && !isLastCol) ? mat_c.readwrite_sender(ij)
                                                 : splitTile(mat_c(ij), {{0, 0}, {nrows, ncols}});

        using dlaf::internal::whenAllLift;
        ex::start_detached(whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, alpha,
                                       panelA.read_sender(ij), panelB.read_sender(ij),
                                       k == idx_begin ? beta : T(1), std::move(tile_c)) |
                           tile::gemm(dlaf::internal::Policy<B>()));
      }
    }

    panelA.reset();
    panelB.reset();
  }
}
}
}
