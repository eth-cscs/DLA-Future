//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cstddef>

#include <dlaf/blas/tile.h>
#include <dlaf/blas/tile_extensions.h>
#include <dlaf/common/assert.h>
#include <dlaf/common/index2d.h>
#include <dlaf/common/round_robin.h>
#include <dlaf/communication/broadcast_panel.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_ref.h>
#include <dlaf/matrix/panel.h>
#include <dlaf/multiplication/general/api.h>
#include <dlaf/sender/when_all_lift.h>
#include <dlaf/util_matrix.h>

namespace dlaf::multiplication::internal {

template <Backend B, Device D, class T>
void General<B, D, T>::callNN(const T alpha, MatrixRef<const T, D>& mat_a, MatrixRef<const T, D>& mat_b,
                              const T beta, MatrixRef<T, D>& mat_c) {
  namespace ex = pika::execution::experimental;

  if (mat_a.nr_tiles().cols() == 0) {
    // Note: if beta == 1, we optimize by not even scheduling anything
    if (beta != T(1)) {
      for (SizeType j = 0; j < mat_c.distribution().local_nr_tiles().cols(); ++j)
        for (SizeType i = 0; i < mat_c.distribution().local_nr_tiles().rows(); ++i)
          ex::start_detached(dlaf::internal::whenAllLift(beta, mat_c.readwrite(LocalTileIndex(i, j))) |
                             tile::scal(dlaf::internal::Policy<B>()));
    }
    return;
  }

  for (SizeType j = 0; j < mat_c.distribution().local_nr_tiles().cols(); ++j) {
    for (SizeType i = 0; i < mat_c.distribution().local_nr_tiles().rows(); ++i) {
      for (SizeType k = 0; k < mat_a.distribution().local_nr_tiles().cols(); ++k) {
        ex::start_detached(dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, alpha,
                                                       mat_a.read(LocalTileIndex(i, k)),
                                                       mat_b.read(LocalTileIndex(k, j)),
                                                       k == 0 ? beta : T(1),
                                                       mat_c.readwrite(LocalTileIndex(i, j))) |
                           tile::gemm(dlaf::internal::Policy<B>()));
      }
    }
  }
}

template <Backend B, Device D, class T>
void General<B, D, T>::callNN(comm::CommunicatorPipeline<comm::CommunicatorType::Row>& row_task_chain,
                              comm::CommunicatorPipeline<comm::CommunicatorType::Col>& col_task_chain,
                              const T alpha, MatrixRef<const T, D>& mat_a, MatrixRef<const T, D>& mat_b,
                              const T beta, MatrixRef<T, D>& mat_c) {
  namespace ex = pika::execution::experimental;

  if (mat_c.size().isEmpty())
    return;

  const matrix::Distribution& dist_a = mat_a.distribution();
  const matrix::Distribution& dist_b = mat_b.distribution();
  const matrix::Distribution& dist_c = mat_c.distribution();
  const auto rank = dist_c.rank_index();

  if (mat_a.nr_tiles().cols() == 0) {
    // Note: if beta == 1, we optimize by not even scheduling anything
    if (beta != T(1)) {
      for (SizeType j = 0; j < mat_c.distribution().local_nr_tiles().cols(); ++j)
        for (SizeType i = 0; i < mat_c.distribution().local_nr_tiles().rows(); ++i)
          ex::start_detached(dlaf::internal::whenAllLift(beta, mat_c.readwrite(LocalTileIndex(i, j))) |
                             tile::scal(dlaf::internal::Policy<B>()));
    }
    return;
  }

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, D>> panelsA(n_workspaces, dist_a);
  common::RoundRobin<matrix::Panel<Coord::Row, T, D>> panelsB(n_workspaces, dist_b);

  DLAF_ASSERT_HEAVY(mat_a.nr_tiles().cols() == mat_b.nr_tiles().rows(), mat_a.nr_tiles(),
                    mat_b.nr_tiles());

  // This loops over the global indices for k, because every rank has to participate in communication
  for (SizeType k = 0; k < mat_a.nr_tiles().cols(); ++k) {
    auto& panelA = panelsA.nextResource();
    auto& panelB = panelsB.nextResource();

    if (k == 0 || k == mat_a.nr_tiles().cols() - 1) {
      DLAF_ASSERT_HEAVY(dist_a.global_tile_size_of<Coord::Col>(k) ==
                            dist_b.global_tile_size_of<Coord::Row>(k),
                        dist_a.global_tile_size_of<Coord::Col>(k),
                        dist_b.global_tile_size_of<Coord::Row>(k));
      const SizeType kSize = dist_a.global_tile_size_of<Coord::Col>(k);
      panelA.setWidth(kSize);
      panelB.setHeight(kSize);
    }

    // Setup the column workspace for the root ranks, i.e. the ones in the current col
    const auto rank_k_col = dist_a.rank_global_tile<Coord::Col>(k);
    if (rank_k_col == rank.col()) {
      const auto k_local = dist_a.local_tile_from_global_tile<Coord::Col>(k);
      for (SizeType i = 0; i < dist_c.local_nr_tiles().rows(); ++i) {
        const LocalTileIndex ik(i, k_local);
        panelA.setTile(ik, mat_a.read(ik));
      }
    }
    // Setup the row workspace for the root ranks, i.e. the ones in the current row
    const auto rank_k_row = dist_b.rank_global_tile<Coord::Row>(k);
    if (rank_k_row == rank.row()) {
      const auto k_local = dist_b.local_tile_from_global_tile<Coord::Row>(k);
      for (SizeType j = 0; j < dist_c.local_nr_tiles().cols(); ++j) {
        const LocalTileIndex kj(k_local, j);
        panelB.setTile(kj, mat_b.read(kj));
      }
    }

    // Broadcast both column and row panel from root to others (row-wise and col-wise, respectively)
    broadcast(rank_k_col, panelA, row_task_chain);
    broadcast(rank_k_row, panelB, col_task_chain);

    // This is the core loop where the k step performs the update over the entire local matrix using
    // the col and row workspaces.
    // Everything needed for the update is available locally thanks to previous broadcasts.
    for (SizeType i = 0; i < dist_c.local_nr_tiles().rows(); ++i) {
      for (SizeType j = 0; j < dist_c.local_nr_tiles().cols(); ++j) {
        const LocalTileIndex ij(i, j);

        ex::start_detached(dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, alpha,
                                                       panelA.read(ij), panelB.read(ij),
                                                       k == 0 ? beta : T(1), mat_c.readwrite(ij)) |
                           tile::gemm(dlaf::internal::Policy<B>()));
      }
    }

    panelA.reset();
    panelB.reset();
  }
}
}
