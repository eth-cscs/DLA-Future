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
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace factorization {
namespace internal {

namespace cholesky_l {
template <class Executor, Device device, class T>
void potrfDiagTile(Executor&& exec, hpx::future<matrix::Tile<T, device>> matrix_tile) {
  hpx::dataflow(exec, matrix::unwrapExtendTiles(tile::potrf_o), blas::Uplo::Lower,
                std::move(matrix_tile));
}

template <class Executor, Device device, class T>
void trsmPanelTile(Executor&& executor_hp, hpx::shared_future<matrix::Tile<const T, device>> kk_tile,
                   hpx::future<matrix::Tile<T, device>> matrix_tile) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), blas::Side::Right,
                blas::Uplo::Lower, blas::Op::ConjTrans, blas::Diag::NonUnit, T(1.0), std::move(kk_tile),
                std::move(matrix_tile));
}

template <class Executor, Device device, class T>
void herkTrailingDiagTile(Executor&& trailing_matrix_executor,
                          hpx::shared_future<matrix::Tile<const T, device>> panel_tile,
                          hpx::future<matrix::Tile<T, device>> matrix_tile) {
  hpx::dataflow(trailing_matrix_executor, matrix::unwrapExtendTiles(tile::herk_o), blas::Uplo::Lower,
                blas::Op::NoTrans, BaseType<T>(-1.0), panel_tile, BaseType<T>(1.0),
                std::move(matrix_tile));
}

template <class Executor, Device device, class T>
void gemmTrailingMatrixTile(Executor&& trailing_matrix_executor,
                            hpx::shared_future<matrix::Tile<const T, device>> panel_tile,
                            hpx::shared_future<matrix::Tile<const T, device>> col_panel,
                            hpx::future<matrix::Tile<T, device>> matrix_tile) {
  hpx::dataflow(trailing_matrix_executor, matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::NoTrans,
                blas::Op::ConjTrans, T(-1.0), std::move(panel_tile), std::move(col_panel), T(1.0),
                std::move(matrix_tile));
}
}

namespace cholesky_u {
template <class Executor, Device device, class T>
void potrfDiagTile(Executor&& exec, hpx::future<matrix::Tile<T, device>> matrix_tile) {
  hpx::dataflow(exec, matrix::unwrapExtendTiles(tile::potrf_o), blas::Uplo::Upper,
                std::move(matrix_tile));
}

template <class Executor, Device device, class T>
void trsmPanelTile(Executor&& executor_hp, hpx::shared_future<matrix::Tile<const T, device>> kk_tile,
                   hpx::future<matrix::Tile<T, device>> matrix_tile) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), blas::Side::Left,
                blas::Uplo::Upper, blas::Op::ConjTrans, blas::Diag::NonUnit, T(1.0), std::move(kk_tile),
                std::move(matrix_tile));
}

template <class Executor, Device device, class T>
void herkTrailingDiagTile(Executor&& trailing_matrix_executor,
                          hpx::shared_future<matrix::Tile<const T, device>> panel_tile,
                          hpx::future<matrix::Tile<T, device>> matrix_tile) {
  hpx::dataflow(trailing_matrix_executor, matrix::unwrapExtendTiles(tile::herk_o), blas::Uplo::Upper,
                blas::Op::ConjTrans, BaseType<T>(-1.0), panel_tile, BaseType<T>(1.0),
                std::move(matrix_tile));
}

template <class Executor, Device device, class T>
void gemmTrailingMatrixTile(Executor&& trailing_matrix_executor,
                            hpx::shared_future<matrix::Tile<const T, device>> panel_tile,
                            hpx::shared_future<matrix::Tile<const T, device>> col_panel,
                            hpx::future<matrix::Tile<T, device>> matrix_tile) {
  hpx::dataflow(trailing_matrix_executor, matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::ConjTrans,
                blas::Op::NoTrans, T(-1.0), std::move(panel_tile), std::move(col_panel), T(1.0),
                std::move(matrix_tile));
}
}

// Local implementation of Lower Cholesky factorization.
template <Backend backend, Device device, class T>
void Cholesky<backend, device, T>::call_L(Matrix<T, device>& mat_a) {
  using namespace cholesky_l;

  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();

  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().cols();

  for (SizeType k = 0; k < nrtile; ++k) {
    // Cholesky decomposition on mat_a(k,k) r/w potrf (lapack operation)
    auto kk = LocalTileIndex{k, k};

    potrfDiagTile(executor_hp, mat_a(kk));

    for (SizeType i = k + 1; i < nrtile; ++i) {
      // Update panel mat_a(i,k) with trsm (blas operation), using data mat_a.read(k,k)
      trsmPanelTile(executor_hp, mat_a.read(kk), mat_a(LocalTileIndex{i, k}));
    }

    for (SizeType j = k + 1; j < nrtile; ++j) {
      // first trailing panel gets high priority (look ahead).
      auto& trailing_matrix_executor = (j == k + 1) ? executor_hp : executor_np;

      // Update trailing matrix: diagonal element mat_a(j,j), reading mat_a.read(j,k), using herk (blas operation)
      herkTrailingDiagTile(trailing_matrix_executor, mat_a.read(LocalTileIndex{j, k}),
                           mat_a(LocalTileIndex{j, j}));

      for (SizeType i = j + 1; i < nrtile; ++i) {
        // Update remaining trailing matrix mat_a(i,j), reading mat_a.read(i,k) and mat_a.read(j,k),
        // using gemm (blas operation)
        gemmTrailingMatrixTile(trailing_matrix_executor, mat_a.read(LocalTileIndex{i, k}),
                               mat_a.read(LocalTileIndex{j, k}), mat_a(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Cholesky<backend, device, T>::call_L(comm::CommunicatorGrid grid, Matrix<T, device>& mat_a) {
  using namespace cholesky_l;

  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();
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
      potrfDiagTile(executor_hp, mat_a(kk_idx));

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

        trsmPanelTile(executor_hp, panelT.read(diag_wp_idx), mat_a(ik_idx));

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
      auto& trailing_matrix_executor = (jt_idx == kt) ? executor_hp : executor_np;
      if (this_rank.row() == owner.row()) {
        const auto i = distr.localTileFromGlobalTile<Coord::Row>(jt_idx);

        herkTrailingDiagTile(trailing_matrix_executor, panel.read({Coord::Row, i}),
                             mat_a(LocalTileIndex{i, j}));
      }

      for (SizeType i_idx = jt_idx + 1; i_idx < nrtile; ++i_idx) {
        const auto owner_row = distr.rankGlobalTile<Coord::Row>(i_idx);

        if (owner_row != this_rank.row())
          continue;

        const auto i = distr.localTileFromGlobalTile<Coord::Row>(i_idx);
        gemmTrailingMatrixTile(executor_np, panel.read({Coord::Row, i}), panelT.read({Coord::Col, j}),
                               mat_a(LocalTileIndex{i, j}));
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

  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();

  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().cols();

  for (SizeType k = 0; k < nrtile; ++k) {
    auto kk = LocalTileIndex{k, k};

    potrfDiagTile(executor_hp, mat_a(kk));

    for (SizeType j = k + 1; j < nrtile; ++j) {
      trsmPanelTile(executor_hp, mat_a.read(kk), mat_a(LocalTileIndex{k, j}));
    }

    for (SizeType i = k + 1; i < nrtile; ++i) {
      auto& trailing_matrix_executor = (i == k + 1) ? executor_hp : executor_np;

      herkTrailingDiagTile(trailing_matrix_executor, mat_a.read(LocalTileIndex{k, i}),
                           mat_a(LocalTileIndex{i, i}));

      for (SizeType j = i + 1; j < nrtile; ++j) {
        gemmTrailingMatrixTile(executor_np, mat_a.read(LocalTileIndex{k, i}),
                               mat_a.read(LocalTileIndex{k, j}), mat_a(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Cholesky<backend, device, T>::call_U(comm::CommunicatorGrid grid, Matrix<T, device>& mat_a) {
  using namespace cholesky_u;

  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();
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
      potrfDiagTile(executor_hp, mat_a(kk_idx));
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

        trsmPanelTile(executor_hp, panelT.read(diag_wp_idx), mat_a(kj_idx));

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

      auto& trailing_matrix_executor = (it_idx == kt) ? executor_hp : executor_np;
      if (this_rank.col() == owner.col()) {
        const auto j = distr.localTileFromGlobalTile<Coord::Col>(it_idx);

        herkTrailingDiagTile(trailing_matrix_executor, panel.read({Coord::Col, j}),
                             mat_a(LocalTileIndex{i, j}));
      }

      for (SizeType j_idx = it_idx + 1; j_idx < nrtile; ++j_idx) {
        const auto owner_col = distr.rankGlobalTile<Coord::Col>(j_idx);

        if (owner_col != this_rank.col())
          continue;

        const auto j = distr.localTileFromGlobalTile<Coord::Col>(j_idx);

        gemmTrailingMatrixTile(executor_np, panelT.read({Coord::Row, i}), panel.read({Coord::Col, j}),
                               mat_a(LocalTileIndex{i, j}));
      }
    }

    panel.reset();
    panelT.reset();
  }
}
/// ---- ETI
#define DLAF_CHOLESKY_MC_ETI(KWORD, DATATYPE) \
  KWORD template struct Cholesky<Backend::MC, Device::CPU, DATATYPE>;

DLAF_CHOLESKY_MC_ETI(extern, float)
DLAF_CHOLESKY_MC_ETI(extern, double)
DLAF_CHOLESKY_MC_ETI(extern, std::complex<float>)
DLAF_CHOLESKY_MC_ETI(extern, std::complex<double>)

}
}
}
