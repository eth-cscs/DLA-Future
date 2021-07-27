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
#include "dlaf/eigensolver/gen_to_std/api.h"
#include "dlaf/executors.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

namespace gentostd_l {
template <class Executor, Device device, class T>
void hegstDiagTile(Executor&& executor_hp, hpx::future<matrix::Tile<T, device>> a_kk,
                   hpx::future<matrix::Tile<T, device>> l_kk) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::hegst_o), 1, blas::Uplo::Lower,
                std::move(a_kk), std::move(l_kk));
}

template <class Executor, Device device, class T>
void trsmPanelTile(Executor&& executor_hp, hpx::shared_future<matrix::Tile<const T, device>> l_kk,
                   hpx::future<matrix::Tile<T, device>> a_ik) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), blas::Side::Right,
                blas::Uplo::Lower, blas::Op::ConjTrans, blas::Diag::NonUnit, T(1.0), l_kk,
                std::move(a_ik));
}

template <class Executor, Device device, class T>
void hemmPanelTile(Executor&& executor_hp, hpx::shared_future<matrix::Tile<const T, device>> a_kk,
                   hpx::shared_future<matrix::Tile<const T, device>> l_ik,
                   hpx::future<matrix::Tile<T, device>> a_ik) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::hemm_o), blas::Side::Right,
                blas::Uplo::Lower, T(-0.5), a_kk, l_ik, T(1.0), std::move(a_ik));
}

template <class Executor, Device device, class T>
void her2kTrailingDiagTile(Executor&& ex, hpx::shared_future<matrix::Tile<const T, device>> a_jk,
                           hpx::shared_future<matrix::Tile<const T, device>> l_jk,
                           hpx::future<matrix::Tile<T, device>> a_kk) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::her2k_o), blas::Uplo::Lower, blas::Op::NoTrans,
                T(-1.0), a_jk, l_jk, BaseType<T>(1.0), std::move(a_kk));
}

template <class Executor, Device device, class T>
void gemmTrailingMatrixTile(Executor&& ex, hpx::shared_future<matrix::Tile<const T, device>> mat_ik,
                            hpx::shared_future<matrix::Tile<const T, device>> mat_jk,
                            hpx::future<matrix::Tile<T, device>> a_ij) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::NoTrans, blas::Op::ConjTrans,
                T(-1.0), mat_ik, mat_jk, T(1.0), std::move(a_ij));
}

template <class Executor, Device device, class T>
void trsmPanelUpdateTile(Executor&& executor_hp, hpx::shared_future<matrix::Tile<const T, device>> l_jj,
                         hpx::future<matrix::Tile<T, device>> a_jk) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), blas::Side::Left,
                blas::Uplo::Lower, blas::Op::NoTrans, blas::Diag::NonUnit, T(1.0), l_jj,
                std::move(a_jk));
}

template <class Executor, Device device, class T>
void gemmPanelUpdateTile(Executor&& ex, hpx::shared_future<matrix::Tile<const T, device>> l_ij,
                         hpx::shared_future<matrix::Tile<const T, device>> a_jk,
                         hpx::future<matrix::Tile<T, device>> a_ik) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::NoTrans, blas::Op::NoTrans,
                T(-1.0), l_ij, a_jk, T(1.0), std::move(a_ik));
}
}

namespace gentostd_u {
template <class Executor, Device device, class T>
void hegstDiagTile(Executor&& executor_hp, hpx::future<matrix::Tile<T, device>> a_kk,
                   hpx::future<matrix::Tile<T, device>> u_kk) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::hegst_o), 1, blas::Uplo::Upper,
                std::move(a_kk), std::move(u_kk));
}

template <class Executor, Device device, class T>
void trsmPanelTile(Executor&& executor_hp, hpx::shared_future<matrix::Tile<const T, device>> u_kk,
                   hpx::future<matrix::Tile<T, device>> a_ki) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), blas::Side::Left,
                blas::Uplo::Upper, blas::Op::ConjTrans, blas::Diag::NonUnit, T(1.0), u_kk,
                std::move(a_ki));
}

template <class Executor, Device device, class T>
void hemmPanelTile(Executor&& executor_hp, hpx::shared_future<matrix::Tile<const T, device>> a_kk,
                   hpx::shared_future<matrix::Tile<const T, device>> u_ki,
                   hpx::future<matrix::Tile<T, device>> a_ki) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::hemm_o), blas::Side::Left,
                blas::Uplo::Upper, T(-0.5), a_kk, u_ki, T(1.0), std::move(a_ki));
}

template <class Executor, Device device, class T>
void her2kTrailingDiagTile(Executor&& ex, hpx::shared_future<matrix::Tile<const T, device>> a_ki,
                           hpx::shared_future<matrix::Tile<const T, device>> u_ki,
                           hpx::future<matrix::Tile<T, device>> a_ii) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::her2k_o), blas::Uplo::Upper, blas::Op::ConjTrans,
                T(-1.0), a_ki, u_ki, BaseType<T>(1.0), std::move(a_ii));
}

template <class Executor, Device device, class T>
void gemmTrailingMatrixTile(Executor&& ex, hpx::shared_future<matrix::Tile<const T, device>> mat_ki,
                            hpx::shared_future<matrix::Tile<const T, device>> mat_kj,
                            hpx::future<matrix::Tile<T, device>> a_ij) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::ConjTrans, blas::Op::NoTrans,
                T(-1.0), mat_ki, mat_kj, T(1.0), std::move(a_ij));
}

template <class Executor, Device device, class T>
void trsmPanelUpdateTile(Executor&& executor_hp, hpx::shared_future<matrix::Tile<const T, device>> u_ii,
                         hpx::future<matrix::Tile<T, device>> a_ki) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), blas::Side::Right,
                blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit, T(1.0), u_ii,
                std::move(a_ki));
}

template <class Executor, Device device, class T>
void gemmPanelUpdateTile(Executor&& ex, hpx::shared_future<matrix::Tile<const T, device>> a_ki,
                         hpx::shared_future<matrix::Tile<const T, device>> u_ij,
                         hpx::future<matrix::Tile<T, device>> a_kj) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::NoTrans, blas::Op::NoTrans,
                T(-1.0), a_ki, u_ij, T(1.0), std::move(a_kj));
}
}

// Implementation based on LAPACK Algorithm for the transformation from generalized to standard
// eigenproblem (xHEGST)
template <Backend backend, Device device, class T>
void GenToStd<backend, device, T>::call_L(Matrix<T, device>& mat_a, Matrix<T, device>& mat_l) {
  using namespace gentostd_l;
  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();

  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().cols();

  for (SizeType k = 0; k < nrtile; ++k) {
    const LocalTileIndex kk{k, k};

    // Direct transformation to standard eigenvalue problem of the diagonal tile
    hegstDiagTile(executor_hp, mat_a(kk), mat_l(kk));

    // If there is no trailing matrix
    if (k == nrtile - 1)
      continue;

    for (SizeType i = k + 1; i < nrtile; ++i) {
      const LocalTileIndex ik{i, k};
      trsmPanelTile(executor_hp, mat_l.read(kk), mat_a(ik));
      hemmPanelTile(executor_hp, mat_a.read(kk), mat_l.read(ik), mat_a(ik));
    }

    for (SizeType j = k + 1; j < nrtile; ++j) {
      const LocalTileIndex jj{j, j};
      const LocalTileIndex jk{j, k};
      // first trailing panel gets high priority (look ahead).
      auto& trailing_matrix_executor = (j == k + 1) ? executor_hp : executor_np;

      her2kTrailingDiagTile(trailing_matrix_executor, mat_a.read(jk), mat_l.read(jk), mat_a(jj));

      for (SizeType i = j + 1; i < nrtile; ++i) {
        const LocalTileIndex ik{i, k};
        const LocalTileIndex ij{i, j};
        gemmTrailingMatrixTile(trailing_matrix_executor, mat_a.read(ik), mat_l.read(jk), mat_a(ij));
        gemmTrailingMatrixTile(trailing_matrix_executor, mat_l.read(ik), mat_a.read(jk), mat_a(ij));
      }
    }

    for (SizeType i = k + 1; i < nrtile; ++i) {
      const LocalTileIndex ik{i, k};
      hemmPanelTile(executor_np, mat_a.read(kk), mat_l.read(ik), mat_a(ik));
    }

    for (SizeType j = k + 1; j < nrtile; ++j) {
      const LocalTileIndex jj{j, j};
      const LocalTileIndex jk{j, k};
      trsmPanelUpdateTile(executor_hp, mat_l.read(jj), mat_a(jk));

      for (SizeType i = j + 1; i < nrtile; ++i) {
        const LocalTileIndex ij{i, j};
        const LocalTileIndex ik{i, k};
        gemmPanelUpdateTile(executor_np, mat_l.read(ij), mat_a.read(jk), mat_a(ik));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void GenToStd<backend, device, T>::call_L(comm::CommunicatorGrid grid, Matrix<T, device>& mat_a,
                                          Matrix<T, device>& mat_l) {
  using namespace gentostd_l;
  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();
  auto executor_mpi = dlaf::getMPIExecutor<backend>();

  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr = mat_a.distribution();
  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().cols();

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> a_panels(n_workspaces, distr);
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> a_panelsT(n_workspaces, distr);
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> l_panels(n_workspaces, distr);
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> l_panelsT(n_workspaces, distr);

  for (SizeType k = 0; k < nrtile; ++k) {
    const GlobalTileIndex kk{k, k};
    const comm::Index2D kk_rank = distr.rankGlobalTile(kk);

    const LocalTileSize kk_offset{
        distr.nextLocalTileFromGlobalTile<Coord::Row>(k),
        distr.nextLocalTileFromGlobalTile<Coord::Col>(k),
    };

    const LocalTileSize at_offset{
        distr.nextLocalTileFromGlobalTile<Coord::Row>(k + 1),
        distr.nextLocalTileFromGlobalTile<Coord::Col>(k + 1),
    };

    auto& l_panel = l_panels.nextResource();
    auto& l_panelT = l_panelsT.nextResource();
    auto& a_panel = a_panels.nextResource();
    auto& a_panelT = a_panelsT.nextResource();
    l_panel.setRangeStart({k, k});
    l_panelT.setRangeStart({k, k});
    a_panelT.setRange({0, 0}, {k, k});

    if (k == nrtile - 1) {
      l_panel.setWidth(distr.tileSize(kk).cols());
      a_panelT.setHeight(distr.tileSize(kk).rows());
    }

    if (kk_rank.col() == this_rank.col()) {
      for (SizeType i_local = kk_offset.rows(); i_local < distr.localNrTiles().rows(); ++i_local) {
        const LocalTileIndex ik_panel(Coord::Row, i_local);
        const LocalTileIndex ik(i_local, kk_offset.cols());
        l_panel.setTile(ik_panel, mat_l.read(ik));
      }
    }

    broadcast(executor_mpi, kk_rank.col(), l_panel, l_panelT, mpi_row_task_chain, mpi_col_task_chain);

    // continue update previous panels
    // Note: The tasks of the final huge TRSM of the HEGST step have been reshuffled to avoid extra
    //       communication of the matrix L.
    //       During k-th iteration only the tasks involving the k-th panel of L are executed.
    //       Therefore, all previous panel have to be updated at each step.
    if (kk_rank.row() == this_rank.row()) {
      for (SizeType j_local = 0; j_local < kk_offset.cols(); ++j_local) {
        const LocalTileIndex kk_panel(Coord::Row, kk_offset.rows());
        const LocalTileIndex kj_panelT{Coord::Col, j_local};
        const LocalTileIndex kj(kk_offset.rows(), j_local);

        trsmPanelUpdateTile(executor_hp, l_panel.read(kk_panel), mat_a(kj));

        a_panelT.setTile(kj_panelT, mat_a.read(kj));
      }
    }

    // No next rows update if last row.
    if (k < nrtile - 1) {
      broadcast(executor_mpi, kk_rank.row(), a_panelT, mpi_col_task_chain);

      for (SizeType j_local = 0; j_local < kk_offset.cols(); ++j_local) {
        for (SizeType i_local = at_offset.rows(); i_local < distr.localNrTiles().rows(); ++i_local) {
          const LocalTileIndex ik_panel{Coord::Row, i_local};
          const LocalTileIndex kj_panelT{Coord::Col, j_local};
          const LocalTileIndex ij{i_local, j_local};

          gemmPanelUpdateTile(executor_np, l_panel.read(ik_panel), a_panelT.read(kj_panelT), mat_a(ij));
        }
      }
    }

    a_panelT.reset();

    // Direct transformation to standard eigenvalue problem of the diagonal tile
    if (kk_rank == this_rank)
      hegstDiagTile(executor_hp, mat_a(kk), mat_l(kk));

    // If there is no trailing matrix
    if (k == nrtile - 1)
      continue;

    const LocalTileIndex diag_wp_idx{0, kk_offset.cols()};

    a_panel.setRangeStart({k + 1, k + 1});

    hpx::shared_future<matrix::Tile<const T, device>> a_diag;
    if (kk_rank.col() == this_rank.col()) {
      // Note:
      // [a,l]_panelT shrinked to a single tile for temporarly storing and communicating the diagonal
      // tile used for the column update
      a_panelT.setRange({k, k}, {k + 1, k + 1});

      if (kk_rank.row() == this_rank.row()) {
        a_panelT.setTile(diag_wp_idx, mat_a.read(kk));
      }
      broadcast(executor_mpi, kk_rank.row(), a_panelT, mpi_col_task_chain);

      // panel partial update
      for (SizeType i_local = at_offset.rows(); i_local < distr.localNrTiles().rows(); ++i_local) {
        const LocalTileIndex ik_panel(Coord::Row, i_local);
        const LocalTileIndex ik(i_local, distr.localTileFromGlobalTile<Coord::Col>(k));

        trsmPanelTile(executor_hp, l_panelT.read(diag_wp_idx), mat_a(ik));
        hemmPanelTile(executor_hp, a_panelT.read(diag_wp_idx), mat_l.read(ik), mat_a(ik));

        // keep diagonal tile for later.
        a_diag = a_panelT.read(diag_wp_idx);

        a_panel.setTile(ik_panel, mat_a.read(ik));
      }

      // row panel has been used for temporary storage of diagonal panel for column update
      a_panelT.reset();
    }

    a_panelT.setRange({k + 1, k + 1}, common::indexFromOrigin(distr.nrTiles()));

    broadcast(executor_mpi, kk_rank.col(), a_panel, a_panelT, mpi_row_task_chain, mpi_col_task_chain);

    // trailing matrix update
    for (SizeType j = k + 1; j < nrtile; ++j) {
      const auto owner = distr.rankGlobalTile({j, j});

      if (owner.col() != this_rank.col())
        continue;

      const auto j_local = distr.localTileFromGlobalTile<Coord::Col>(j);
      // first trailing panel gets high priority (look ahead).
      auto& trailing_matrix_executor = (j == k + 1) ? executor_hp : executor_np;
      if (this_rank.row() == owner.row()) {
        const auto i_local = distr.localTileFromGlobalTile<Coord::Row>(j);

        her2kTrailingDiagTile(trailing_matrix_executor, a_panel.read({Coord::Row, i_local}),
                              l_panel.read({Coord::Row, i_local}),
                              mat_a(LocalTileIndex{i_local, j_local}));
      }

      for (SizeType i = j + 1; i < nrtile; ++i) {
        const auto owner_row = distr.rankGlobalTile<Coord::Row>(i);

        if (owner_row != this_rank.row())
          continue;

        const auto i_local = distr.localTileFromGlobalTile<Coord::Row>(i);
        const LocalTileIndex ik_panel{Coord::Row, i_local};
        const LocalTileIndex kj_panelT{Coord::Col, j_local};
        const LocalTileIndex ij{i_local, j_local};

        gemmTrailingMatrixTile(executor_np, a_panel.read(ik_panel), l_panelT.read(kj_panelT), mat_a(ij));
        gemmTrailingMatrixTile(executor_np, l_panel.read(ik_panel), a_panelT.read(kj_panelT), mat_a(ij));
      }
    }

    a_panel.reset();
    a_panelT.reset();
    l_panel.reset();
    l_panelT.reset();

    if (kk_rank.col() == this_rank.col()) {
      // panel partial update
      for (SizeType i_local = at_offset.rows(); i_local < distr.localNrTiles().rows(); ++i_local) {
        const LocalTileIndex local_idx(Coord::Row, i_local);
        const LocalTileIndex ik(i_local, distr.localTileFromGlobalTile<Coord::Col>(k));

        hemmPanelTile(executor_hp, a_diag, mat_l.read(ik), mat_a(ik));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void GenToStd<backend, device, T>::call_U(Matrix<T, device>& mat_a, Matrix<T, device>& mat_u) {
  using namespace gentostd_u;
  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();

  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().cols();

  for (SizeType k = 0; k < nrtile; ++k) {
    const LocalTileIndex kk{k, k};

    // Direct transformation to standard eigenvalue problem of the diagonal tile
    hegstDiagTile(executor_hp, mat_a(kk), mat_u(kk));

    // If there is no trailing matrix
    if (k == nrtile - 1)
      continue;

    for (SizeType i = k + 1; i < nrtile; ++i) {
      const LocalTileIndex ki{k, i};
      trsmPanelTile(executor_hp, mat_u.read(kk), mat_a(ki));
      hemmPanelTile(executor_hp, mat_a.read(kk), mat_u.read(ki), mat_a(ki));
    }

    for (SizeType i = k + 1; i < nrtile; ++i) {
      const LocalTileIndex ii{i, i};
      const LocalTileIndex ki{k, i};
      // first trailing panel gets high priority (look ahead).
      auto& trailing_matrix_executor = (i == k + 1) ? executor_hp : executor_np;

      her2kTrailingDiagTile(trailing_matrix_executor, mat_a.read(ki), mat_u.read(ki), mat_a(ii));

      for (SizeType j = i + 1; j < nrtile; ++j) {
        const LocalTileIndex kj{k, j};
        const LocalTileIndex ij{i, j};
        gemmTrailingMatrixTile(trailing_matrix_executor, mat_a.read(ki), mat_u.read(kj), mat_a(ij));
        gemmTrailingMatrixTile(trailing_matrix_executor, mat_u.read(ki), mat_a.read(kj), mat_a(ij));
      }
    }

    for (SizeType i = k + 1; i < nrtile; ++i) {
      const LocalTileIndex ki{k, i};
      hemmPanelTile(executor_np, mat_a.read(kk), mat_u.read(ki), mat_a(ki));
    }

    for (SizeType i = k + 1; i < nrtile; ++i) {
      const LocalTileIndex ii{i, i};
      const LocalTileIndex ki{k, i};
      trsmPanelUpdateTile(executor_hp, mat_u.read(ii), mat_a(ki));

      for (SizeType j = i + 1; j < nrtile; ++j) {
        const LocalTileIndex ij{i, j};
        const LocalTileIndex kj{k, j};
        gemmPanelUpdateTile(executor_np, mat_a.read(ki), mat_u.read(ij), mat_a(kj));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void GenToStd<backend, device, T>::call_U(comm::CommunicatorGrid grid, Matrix<T, device>& mat_a,
                                          Matrix<T, device>& mat_u) {
  using namespace gentostd_u;
  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();
  auto executor_mpi = dlaf::getMPIExecutor<backend>();

  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr = mat_a.distribution();
  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().rows();

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> a_panels(n_workspaces, distr);
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> a_panelsT(n_workspaces, distr);
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> u_panels(n_workspaces, distr);
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> u_panelsT(n_workspaces, distr);

  for (SizeType k = 0; k < nrtile; ++k) {
    const GlobalTileIndex kk{k, k};
    const comm::Index2D kk_rank = distr.rankGlobalTile(kk);
    const auto kt = k + 1;

    const LocalTileSize kk_offset{
        distr.nextLocalTileFromGlobalTile<Coord::Row>(k),
        distr.nextLocalTileFromGlobalTile<Coord::Col>(k),
    };

    const LocalTileSize at_offset{
        distr.nextLocalTileFromGlobalTile<Coord::Row>(kt),
        distr.nextLocalTileFromGlobalTile<Coord::Col>(kt),
    };

    auto& u_panel = u_panels.nextResource();
    auto& u_panelT = u_panelsT.nextResource();
    auto& a_panel = a_panels.nextResource();
    auto& a_panelT = a_panelsT.nextResource();
    u_panel.setRangeStart({k, k});
    u_panelT.setRangeStart({k, k});
    a_panelT.setRange({0, 0}, {k, k});

    if (k == nrtile - 1) {
      u_panel.setHeight(distr.tileSize(kk).rows());
      a_panelT.setWidth(distr.tileSize(kk).cols());
    }

    if (kk_rank.row() == this_rank.row()) {
      for (SizeType i_local = kk_offset.cols(); i_local < distr.localNrTiles().cols(); ++i_local) {
        const LocalTileIndex ki_panel(Coord::Col, i_local);
        const LocalTileIndex ki(kk_offset.rows(), i_local);

        u_panel.setTile(ki_panel, mat_u.read(ki));
      }
    }

    broadcast(executor_mpi, kk_rank.row(), u_panel, u_panelT, mpi_row_task_chain, mpi_col_task_chain);

    // TODO: CHECK continue update previous panels
    // Note: The tasks of the final huge TRSM of the HEGST step have been reshuffled to avoid extra
    //       communication of the matrix L.
    //       During k-th iteration only the tasks involving the k-th panel of L are executed.
    //       Therefore, all previous panel have to be updated at each step.
    if (kk_rank.col() == this_rank.col()) {
      for (SizeType i_local = 0; i_local < kk_offset.rows(); ++i_local) {
        const LocalTileIndex kk_panel(Coord::Col, kk_offset.cols());
        const LocalTileIndex ki_panelT{Coord::Row, i_local};
        const LocalTileIndex ik(i_local, kk_offset.cols());

        trsmPanelUpdateTile(executor_hp, u_panel.read(kk_panel), mat_a(ik));

        a_panelT.setTile(ki_panelT, mat_a.read(ik));
      }
    }

    // No next rows update if last row.
    if (k < nrtile - 1) {
      broadcast(executor_mpi, kk_rank.col(), a_panelT, mpi_row_task_chain);

      for (SizeType i_local = 0; i_local < kk_offset.rows(); ++i_local) {
        for (SizeType j_local = at_offset.cols(); j_local < distr.localNrTiles().cols(); ++j_local) {
          const LocalTileIndex kj_panel{Coord::Col, j_local};
          const LocalTileIndex ik_panelT{Coord::Row, i_local};
          const LocalTileIndex ij{i_local, j_local};

          gemmPanelUpdateTile(executor_np, a_panelT.read(ik_panelT), u_panel.read(kj_panel), mat_a(ij));
        }
      }
    }

    a_panelT.reset();

    // Direct transformation to standard eigenvalue problem of the diagonal tile
    if (kk_rank == this_rank)
      hegstDiagTile(executor_hp, mat_a(kk), mat_u(kk));

    // If there is no trailing matrix
    if (k == nrtile - 1)
      continue;

    const LocalTileIndex diag_wp_idx{kk_offset.rows(), 0};

    a_panel.setRangeStart({kt, kt});

    hpx::shared_future<matrix::Tile<const T, device>> a_diag;
    if (kk_rank.row() == this_rank.row()) {
      // Note:
      // [a,u]_panelT shrinked to a single tile for temporarly storing and communicating the diagonal
      // tile used for the column update
      a_panelT.setRange({k, k}, {kt, kt});

      if (kk_rank.col() == this_rank.col()) {
        a_panelT.setTile(diag_wp_idx, mat_a.read(kk));
      }
      broadcast(executor_mpi, kk_rank.col(), a_panelT, mpi_row_task_chain);

      // panel partial update
      for (SizeType j_local = at_offset.cols(); j_local < distr.localNrTiles().cols(); ++j_local) {
        const LocalTileIndex kj_panel(Coord::Col, j_local);
        const LocalTileIndex kj(distr.localTileFromGlobalTile<Coord::Row>(k), j_local);

        trsmPanelTile(executor_hp, u_panelT.read(diag_wp_idx), mat_a(kj));
        hemmPanelTile(executor_hp, a_panelT.read(diag_wp_idx), mat_u.read(kj), mat_a(kj));

        // keep diagonal tile for later.
        a_diag = a_panelT.read(diag_wp_idx);

        a_panel.setTile(kj_panel, mat_a.read(kj));
      }

      // col panel has been used for temporary storage of diagonal panel for column update
      a_panelT.reset();
    }

    a_panelT.setRange({kt, kt}, common::indexFromOrigin(distr.nrTiles()));

    broadcast(executor_mpi, kk_rank.row(), a_panel, a_panelT, mpi_row_task_chain, mpi_col_task_chain);

    // trailing matrix update
    for (SizeType i = kt; i < nrtile; ++i) {
      const auto owner = distr.rankGlobalTile({i, i});

      if (owner.row() != this_rank.row())
        continue;

      const auto i_local = distr.localTileFromGlobalTile<Coord::Row>(i);
      // first trailing panel gets high priority (look ahead).
      auto& trailing_matrix_executor = (i == kt) ? executor_hp : executor_np;
      if (this_rank.col() == owner.col()) {
        const auto j_local = distr.localTileFromGlobalTile<Coord::Col>(i);

        her2kTrailingDiagTile(trailing_matrix_executor, a_panel.read({Coord::Col, j_local}),
                              u_panel.read({Coord::Col, j_local}),
                              mat_a(LocalTileIndex{i_local, j_local}));
      }

      for (SizeType j = i + 1; j < nrtile; ++j) {
        const auto owner_col = distr.rankGlobalTile<Coord::Col>(j);

        if (owner_col != this_rank.col())
          continue;

        const auto j_local = distr.localTileFromGlobalTile<Coord::Col>(j);
        const auto i_col = distr.localTileFromGlobalTile<Coord::Col>(i);
        const auto j_row = distr.localTileFromGlobalTile<Coord::Row>(j);
        const LocalTileIndex ki_panel{Coord::Row, i_local};
        const LocalTileIndex kj_panelT{Coord::Col, j_local};
        const LocalTileIndex ij{i_local, j_local};

        gemmTrailingMatrixTile(executor_np, a_panelT.read(ki_panel), u_panel.read(kj_panelT), mat_a(ij));
        gemmTrailingMatrixTile(executor_np, u_panelT.read(ki_panel), a_panel.read(kj_panelT), mat_a(ij));
      }
    }

    a_panel.reset();
    a_panelT.reset();
    u_panel.reset();
    u_panelT.reset();

    if (kk_rank.row() == this_rank.row()) {
      // panel partial update
      for (SizeType j_local = at_offset.cols(); j_local < distr.localNrTiles().cols(); ++j_local) {
        const LocalTileIndex local_idx(Coord::Col, j_local);
        const LocalTileIndex ki(distr.localTileFromGlobalTile<Coord::Row>(k), j_local);

        hemmPanelTile(executor_hp, a_diag, mat_u.read(ki), mat_a(ki));
      }
    }
  }
}

}
}
}
