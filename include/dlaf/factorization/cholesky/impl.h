//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <hpx/local/future.hpp>
#include <hpx/local/unwrap.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/communication/broadcast_panel.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/executors.h"
#include "dlaf/factorization/cholesky/api.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/traits.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace factorization {
namespace internal {

namespace cholesky_l {
template <Backend backend, class MatrixTileSender>
void potrfDiagTile(hpx::threads::thread_priority priority, MatrixTileSender&& matrix_tile) {
  dlaf::internal::whenAllLift(blas::Uplo::Lower, std::forward<MatrixTileSender>(matrix_tile)) |
      tile::potrf(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}

template <Backend backend, class KKTileSender, class MatrixTileSender>
void trsmPanelTile(hpx::threads::thread_priority priority, KKTileSender&& kk_tile,
                   MatrixTileSender&& matrix_tile) {
  using ElementType = dlaf::internal::SenderElementType<KKTileSender>;

  dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Lower, blas::Op::ConjTrans,
                              blas::Diag::NonUnit, ElementType(1.0), std::forward<KKTileSender>(kk_tile),
                              std::forward<MatrixTileSender>(matrix_tile)) |
      tile::trsm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}

template <Backend backend, class PanelTileSender, class MatrixTileSender>
void herkTrailingDiagTile(hpx::threads::thread_priority priority, PanelTileSender&& panel_tile,
                          MatrixTileSender&& matrix_tile) {
  using BaseElementType = BaseType<dlaf::internal::SenderElementType<PanelTileSender>>;

  dlaf::internal::whenAllLift(blas::Uplo::Lower, blas::Op::NoTrans, BaseElementType(-1.0),
                              std::forward<PanelTileSender>(panel_tile), BaseElementType(1.0),
                              std::forward<MatrixTileSender>(matrix_tile)) |
      tile::herk(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}

template <Backend backend, class PanelTileSender, class ColPanelSender, class MatrixTileSender>
void gemmTrailingMatrixTile(hpx::threads::thread_priority priority, PanelTileSender&& panel_tile,
                            ColPanelSender&& col_panel, MatrixTileSender&& matrix_tile) {
  using ElementType = dlaf::internal::SenderElementType<PanelTileSender>;

  dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::ConjTrans, ElementType(-1.0),
                              std::forward<PanelTileSender>(panel_tile),
                              std::forward<ColPanelSender>(col_panel), ElementType(1.0),
                              std::forward<MatrixTileSender>(matrix_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}
}

namespace cholesky_u {
template <Backend backend, class MatrixTileSender>
void potrfDiagTile(hpx::threads::thread_priority priority, MatrixTileSender&& matrix_tile) {
  dlaf::internal::whenAllLift(blas::Uplo::Upper, std::forward<MatrixTileSender>(matrix_tile)) |
      tile::potrf(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}

template <Backend backend, class KKTileSender, class MatrixTileSender>
void trsmPanelTile(hpx::threads::thread_priority priority, KKTileSender&& kk_tile,
                   MatrixTileSender&& matrix_tile) {
  using ElementType = dlaf::internal::SenderElementType<KKTileSender>;

  dlaf::internal::whenAllLift(blas::Side::Left, blas::Uplo::Upper, blas::Op::ConjTrans,
                              blas::Diag::NonUnit, ElementType(1.0), std::forward<KKTileSender>(kk_tile),
                              std::forward<MatrixTileSender>(matrix_tile)) |
      tile::trsm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}

template <Backend backend, class PanelTileSender, class MatrixTileSender>
void herkTrailingDiagTile(hpx::threads::thread_priority priority, PanelTileSender&& panel_tile,
                          MatrixTileSender&& matrix_tile) {
  using base_element_type = BaseType<dlaf::internal::SenderElementType<PanelTileSender>>;

  dlaf::internal::whenAllLift(blas::Uplo::Upper, blas::Op::ConjTrans, base_element_type(-1.0),
                              std::forward<PanelTileSender>(panel_tile), base_element_type(1.0),
                              std::forward<MatrixTileSender>(matrix_tile)) |
      tile::herk(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}

template <Backend backend, class PanelTileSender, class ColPanelSender, class MatrixTileSender>
void gemmTrailingMatrixTile(hpx::threads::thread_priority priority, PanelTileSender&& panel_tile,
                            ColPanelSender&& col_panel, MatrixTileSender&& matrix_tile) {
  using ElementType = dlaf::internal::SenderElementType<PanelTileSender>;

  dlaf::internal::whenAllLift(blas::Op::ConjTrans, blas::Op::NoTrans, ElementType(-1.0),
                              std::forward<PanelTileSender>(panel_tile),
                              std::forward<ColPanelSender>(col_panel), ElementType(1.0),
                              std::forward<MatrixTileSender>(matrix_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}
}

// Local implementation of Lower Cholesky factorization.
template <Backend backend, Device device, class T>
void Cholesky<backend, device, T>::call_L(Matrix<T, device>& mat_a) {
  using namespace cholesky_l;
  using hpx::threads::thread_priority;

  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().cols();

  for (SizeType k = 0; k < nrtile; ++k) {
    // Cholesky decomposition on mat_a(k,k) r/w potrf (lapack operation)
    auto kk = LocalTileIndex{k, k};

    potrfDiagTile<backend>(thread_priority::normal, mat_a.readwrite_sender(kk));

    for (SizeType i = k + 1; i < nrtile; ++i) {
      // Update panel mat_a(i,k) with trsm (blas operation), using data mat_a.read(k,k)
      trsmPanelTile<backend>(thread_priority::high, mat_a.read_sender(kk),
                             mat_a.readwrite_sender(LocalTileIndex{i, k}));
    }

    for (SizeType j = k + 1; j < nrtile; ++j) {
      // first trailing panel gets high priority (look ahead).
      const auto trailing_matrix_priority =
          (j == k + 1) ? thread_priority::high : thread_priority::normal;

      // Update trailing matrix: diagonal element mat_a(j,j), reading mat_a.read(j,k), using herk (blas operation)
      herkTrailingDiagTile<backend>(trailing_matrix_priority, mat_a.read_sender(LocalTileIndex{j, k}),
                                    mat_a.readwrite_sender(LocalTileIndex{j, j}));

      for (SizeType i = j + 1; i < nrtile; ++i) {
        // Update remaining trailing matrix mat_a(i,j), reading mat_a.read(i,k) and mat_a.read(j,k),
        // using gemm (blas operation)
        gemmTrailingMatrixTile<backend>(thread_priority::normal, mat_a.read_sender(LocalTileIndex{i, k}),
                                        mat_a.read_sender(LocalTileIndex{j, k}),
                                        mat_a.readwrite_sender(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Cholesky<backend, device, T>::call_L(comm::CommunicatorGrid grid, Matrix<T, device>& mat_a) {
  using namespace cholesky_l;
  using hpx::threads::thread_priority;

  auto executor_mpi = dlaf::getMPIExecutor<backend>();

  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr = mat_a.distribution();
  const SizeType nrtile = mat_a.nrTiles().cols();

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> panels(n_workspaces, distr);
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> panelsT(n_workspaces, distr);

  for (SizeType k = 0; k < nrtile; ++k) {
    const GlobalTileIndex kk_idx(k, k);
    const comm::Index2D kk_rank = distr.rankGlobalTile(kk_idx);

    // Factorization of diagonal tile and broadcast it along the k-th column
    if (kk_rank == this_rank)
      potrfDiagTile<backend>(thread_priority::normal, mat_a.readwrite_sender(kk_idx));

    // If there is no trailing matrix
    const SizeType kt = k + 1;
    if (kt == nrtile)
      continue;

    auto& panel = panels.nextResource();
    auto& panelT = panelsT.nextResource();

    panel.setRangeStart({kt, kt});

    if (kk_rank.col() == this_rank.col()) {
      const LocalTileIndex diag_wp_idx{0, distr.localTileFromGlobalTile<Coord::Col>(k)};

      // Note:
      // panelT shrinked to a single tile for temporarly storing and communicating the diagonal
      // tile used for the column update
      panelT.setRange({k, k}, {kt, kt});

      if (kk_rank.row() == this_rank.row())
        panelT.setTile(diag_wp_idx, mat_a.read(kk_idx));
      broadcast(executor_mpi, kk_rank.row(), panelT, mpi_col_task_chain);

      // COLUMN UPDATE
      for (SizeType i = distr.nextLocalTileFromGlobalTile<Coord::Row>(kt);
           i < distr.localNrTiles().rows(); ++i) {
        const LocalTileIndex local_idx(Coord::Row, i);
        const LocalTileIndex ik_idx(i, distr.localTileFromGlobalTile<Coord::Col>(k));

        trsmPanelTile<backend>(thread_priority::high, panelT.read_sender(diag_wp_idx),
                               mat_a.readwrite_sender(ik_idx));

        panel.setTile(local_idx, mat_a.read(ik_idx));
      }

      // row panel has been used for temporary storage of diagonal panel for column update
      panelT.reset();
    }

    panelT.setRange({kt, kt}, indexFromOrigin(distr.nrTiles()));

    broadcast(executor_mpi, kk_rank.col(), panel, panelT, mpi_row_task_chain, mpi_col_task_chain);

    // TRAILING MATRIX
    for (SizeType jt_idx = kt; jt_idx < nrtile; ++jt_idx) {
      const auto owner = distr.rankGlobalTile({jt_idx, jt_idx});

      if (owner.col() != this_rank.col())
        continue;

      const auto j = distr.localTileFromGlobalTile<Coord::Col>(jt_idx);
      const auto trailing_matrix_priority =
          (jt_idx == kt) ? thread_priority::high : thread_priority::normal;
      if (this_rank.row() == owner.row()) {
        const auto i = distr.localTileFromGlobalTile<Coord::Row>(jt_idx);

        herkTrailingDiagTile<backend>(trailing_matrix_priority, panel.read_sender({Coord::Row, i}),
                                      mat_a.readwrite_sender(LocalTileIndex{i, j}));
      }

      for (SizeType i_idx = jt_idx + 1; i_idx < nrtile; ++i_idx) {
        const auto owner_row = distr.rankGlobalTile<Coord::Row>(i_idx);

        if (owner_row != this_rank.row())
          continue;

        const auto i = distr.localTileFromGlobalTile<Coord::Row>(i_idx);
        // TODO: This was using executor_np. Was that intentional, or should it
        // be trailing_matrix_executor/priority?
        gemmTrailingMatrixTile<backend>(thread_priority::normal, panel.read_sender({Coord::Row, i}),
                                        panelT.read_sender({Coord::Col, j}),
                                        mat_a.readwrite_sender(LocalTileIndex{i, j}));
      }
    }

    panel.reset();
    panelT.reset();
  }
}

// Local implementation of Upper Cholesky factorization.
template <Backend backend, Device device, class T>
void Cholesky<backend, device, T>::call_U(Matrix<T, device>& mat_a) {
  using namespace cholesky_u;
  using hpx::threads::thread_priority;

  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().cols();

  for (SizeType k = 0; k < nrtile; ++k) {
    auto kk = LocalTileIndex{k, k};

    potrfDiagTile<backend>(thread_priority::normal, mat_a.readwrite_sender(kk));

    for (SizeType j = k + 1; j < nrtile; ++j) {
      trsmPanelTile<backend>(thread_priority::high, mat_a.read_sender(kk),
                             mat_a.readwrite_sender(LocalTileIndex{k, j}));
    }

    for (SizeType i = k + 1; i < nrtile; ++i) {
      const auto trailing_matrix_priority =
          (i == k + 1) ? thread_priority::high : thread_priority::normal;

      herkTrailingDiagTile<backend>(trailing_matrix_priority, mat_a.read_sender(LocalTileIndex{k, i}),
                                    mat_a.readwrite_sender(LocalTileIndex{i, i}));

      for (SizeType j = i + 1; j < nrtile; ++j) {
        gemmTrailingMatrixTile<backend>(thread_priority::normal, mat_a.read_sender(LocalTileIndex{k, i}),
                                        mat_a.read_sender(LocalTileIndex{k, j}),
                                        mat_a.readwrite_sender(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Cholesky<backend, device, T>::call_U(comm::CommunicatorGrid grid, Matrix<T, device>& mat_a) {
  using namespace cholesky_u;
  using hpx::threads::thread_priority;

  auto executor_mpi = dlaf::getMPIExecutor<backend>();

  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr = mat_a.distribution();
  const SizeType nrtile = mat_a.nrTiles().cols();

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> panels(n_workspaces, distr);
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> panelsT(n_workspaces, distr);

  for (SizeType k = 0; k < nrtile; ++k) {
    const GlobalTileIndex kk_idx(k, k);
    const comm::Index2D kk_rank = distr.rankGlobalTile(kk_idx);

    // Factorization of diagonal tile and broadcast it along the k-th column
    if (kk_rank == this_rank) {
      potrfDiagTile<backend>(thread_priority::normal, mat_a(kk_idx));
    }

    // If there is no trailing matrix
    const SizeType kt = k + 1;
    if (kt == nrtile)
      continue;

    auto& panel = panels.nextResource();
    auto& panelT = panelsT.nextResource();

    panel.setRangeStart({kt, kt});

    if (kk_rank.row() == this_rank.row()) {
      const LocalTileIndex diag_wp_idx{distr.localTileFromGlobalTile<Coord::Row>(k), 0};
      // Note:
      // panel shrinked to a single tile for temporarly storing and communicating the diagonal
      // tile used for the row update
      panelT.setRange({k, k}, {kt, kt});

      if (kk_rank.col() == this_rank.col())
        panelT.setTile(diag_wp_idx, mat_a.read(kk_idx));
      broadcast(executor_mpi, kk_rank.col(), panelT, mpi_row_task_chain);

      // ROW UPDATE
      for (SizeType j = distr.nextLocalTileFromGlobalTile<Coord::Col>(k + 1);
           j < distr.localNrTiles().cols(); ++j) {
        const LocalTileIndex local_idx(Coord::Col, j);
        const LocalTileIndex kj_idx(distr.localTileFromGlobalTile<Coord::Row>(k), j);

        trsmPanelTile<backend>(thread_priority::high, panelT.read_sender(diag_wp_idx),
                               mat_a.readwrite_sender(kj_idx));

        panel.setTile(local_idx, mat_a.read(kj_idx));
      }

      // col panel has been used for temporary storage of diagonal panel for column update
      panelT.reset();
    }

    panelT.setRange({kt, kt}, indexFromOrigin(distr.nrTiles()));

    broadcast(executor_mpi, kk_rank.row(), panel, panelT, mpi_row_task_chain, mpi_col_task_chain);

    // TRAILING MATRIX
    for (SizeType it_idx = kt; it_idx < nrtile; ++it_idx) {
      const auto owner = distr.rankGlobalTile({it_idx, it_idx});

      if (owner.row() != this_rank.row())
        continue;

      const auto i = distr.localTileFromGlobalTile<Coord::Row>(it_idx);
      const auto trailing_matrix_priority =
          (i == k + 1) ? thread_priority::high : thread_priority::normal;
      if (this_rank.col() == owner.col()) {
        const auto j = distr.localTileFromGlobalTile<Coord::Col>(it_idx);

        herkTrailingDiagTile<backend>(trailing_matrix_priority, panel.read_sender({Coord::Col, j}),
                                      mat_a.readwrite_sender(LocalTileIndex{i, j}));
      }

      for (SizeType j_idx = it_idx + 1; j_idx < nrtile; ++j_idx) {
        const auto owner_col = distr.rankGlobalTile<Coord::Col>(j_idx);

        if (owner_col != this_rank.col())
          continue;

        const auto j = distr.localTileFromGlobalTile<Coord::Col>(j_idx);

        gemmTrailingMatrixTile<backend>(thread_priority::normal, panelT.read_sender({Coord::Row, i}),
                                        panel.read_sender({Coord::Col, j}),
                                        mat_a.readwrite_sender(LocalTileIndex{i, j}));
      }
    }

    panel.reset();
    panelT.reset();
  }
}
}
}
}
