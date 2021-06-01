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

#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/broadcast_panel.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/eigensolver/gen_to_std/api.h"
#include "dlaf/executors.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

template <class T>
void hegstDiagTile(hpx::execution::parallel_executor executor_hp,
                   hpx::future<matrix::Tile<T, Device::CPU>> a_kk,
                   hpx::future<matrix::Tile<T, Device::CPU>> l_kk) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::hegst_o), 1, blas::Uplo::Lower,
                std::move(a_kk), std::move(l_kk));
}

template <class T>
void trsmPanelTile(hpx::execution::parallel_executor executor_hp,
                   hpx::shared_future<matrix::Tile<const T, Device::CPU>> l_kk,
                   hpx::future<matrix::Tile<T, Device::CPU>> a_ik) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), blas::Side::Right,
                blas::Uplo::Lower, blas::Op::ConjTrans, blas::Diag::NonUnit, T(1.0), l_kk,
                std::move(a_ik));
}

template <class T>
void hemmPanelTile(hpx::execution::parallel_executor executor_hp,
                   hpx::shared_future<matrix::Tile<const T, Device::CPU>> a_kk,
                   hpx::shared_future<matrix::Tile<const T, Device::CPU>> l_ik,
                   hpx::future<matrix::Tile<T, Device::CPU>> a_ik) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::hemm_o), blas::Side::Right,
                blas::Uplo::Lower, T(-0.5), a_kk, l_ik, T(1.0), std::move(a_ik));
}

template <class T>
void her2kTrailingDiagTile(hpx::execution::parallel_executor ex,
                           hpx::shared_future<matrix::Tile<const T, Device::CPU>> a_jk,
                           hpx::shared_future<matrix::Tile<const T, Device::CPU>> l_jk,
                           hpx::future<matrix::Tile<T, Device::CPU>> a_kk) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::her2k_o), blas::Uplo::Lower, blas::Op::NoTrans,
                T(-1.0), a_jk, l_jk, BaseType<T>(1.0), std::move(a_kk));
}

template <class T>
void gemmTrailingMatrixTile(hpx::execution::parallel_executor ex,
                            hpx::shared_future<matrix::Tile<const T, Device::CPU>> mat_ik,
                            hpx::shared_future<matrix::Tile<const T, Device::CPU>> mat_jk,
                            hpx::future<matrix::Tile<T, Device::CPU>> a_ij) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::NoTrans, blas::Op::ConjTrans,
                T(-1.0), mat_ik, mat_jk, T(1.0), std::move(a_ij));
}

template <class T>
void trsmPanelUpdateTile(hpx::execution::parallel_executor executor_hp,
                         hpx::shared_future<matrix::Tile<const T, Device::CPU>> l_jj,
                         hpx::future<matrix::Tile<T, Device::CPU>> a_jk) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), blas::Side::Left,
                blas::Uplo::Lower, blas::Op::NoTrans, blas::Diag::NonUnit, T(1.0), l_jj,
                std::move(a_jk));
}

template <class T>
void gemmPanelUpdateTile(hpx::execution::parallel_executor ex,
                         hpx::shared_future<matrix::Tile<const T, Device::CPU>> l_ij,
                         hpx::shared_future<matrix::Tile<const T, Device::CPU>> a_jk,
                         hpx::future<matrix::Tile<T, Device::CPU>> a_ik) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::NoTrans, blas::Op::NoTrans,
                T(-1.0), l_ij, a_jk, T(1.0), std::move(a_ik));
}

// Implementation based on LAPACK Algorithm for the transformation from generalized to standard
// eigenproblem (xHEGST)
template <class T>
struct GenToStd<Backend::MC, Device::CPU, T> {
  static void call_L(Matrix<T, Device::CPU>& mat_a, Matrix<T, Device::CPU>& mat_l);
  static void call_L(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_a,
                     Matrix<T, Device::CPU>& mat_l);
};

template <class T>
void GenToStd<Backend::MC, Device::CPU, T>::call_L(Matrix<T, Device::CPU>& mat_a,
                                                   Matrix<T, Device::CPU>& mat_l) {
  auto executor_hp = dlaf::getHpExecutor<Backend::MC>();
  auto executor_np = dlaf::getNpExecutor<Backend::MC>();

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
      const LocalTileIndex jk{j, k};
      // first trailing panel gets high priority (look ahead).
      auto& trailing_matrix_executor = (j == k + 1) ? executor_hp : executor_np;

      her2kTrailingDiagTile(trailing_matrix_executor, mat_a.read(jk), mat_l.read(jk),
                            mat_a(LocalTileIndex{j, j}));

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
      trsmPanelUpdateTile(executor_hp, mat_l.read(LocalTileIndex{j, j}), mat_a(LocalTileIndex{j, k}));

      for (SizeType i = j + 1; i < nrtile; ++i) {
        gemmPanelUpdateTile(executor_np, mat_l.read(LocalTileIndex{i, j}),
                            mat_a.read(LocalTileIndex{j, k}), mat_a(LocalTileIndex{i, k}));
      }
    }
  }
}

template <class T>
void GenToStd<Backend::MC, Device::CPU, T>::call_L(comm::CommunicatorGrid grid,
                                                   Matrix<T, Device::CPU>& mat_a,
                                                   Matrix<T, Device::CPU>& mat_l) {
  auto executor_hp = dlaf::getHpExecutor<Backend::MC>();
  auto executor_np = dlaf::getNpExecutor<Backend::MC>();

  // Set up MPI executor pipelines
  comm::Executor executor_mpi;
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr = mat_a.distribution();
  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().cols();

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> a_panels(n_workspaces, distr);
  common::RoundRobin<matrix::Panel<Coord::Row, T, Device::CPU>> a_panelsT(n_workspaces, distr);
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> l_panels(n_workspaces, distr);
  common::RoundRobin<matrix::Panel<Coord::Row, T, Device::CPU>> l_panelsT(n_workspaces, distr);

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
    l_panel.setRangeStart(kk_offset);
    l_panelT.setRangeStart(kk_offset);

    // TODO: Need incomplete panel to simplify the code.
    if (k < nrtile - 1) {
      if (kk_rank.col() == this_rank.col()) {
        for (SizeType i_local = kk_offset.rows(); i_local < distr.localNrTiles().rows(); ++i_local) {
          const LocalTileIndex ik_panel(Coord::Row, i_local);
          const LocalTileIndex ik(i_local, kk_offset.cols());
          l_panel.setTile(ik_panel, mat_l.read(ik));
        }
      }

      // TODO skip last step tile
      broadcast(executor_mpi, kk_rank.col(), l_panel, l_panelT, mpi_row_task_chain, mpi_col_task_chain);

      a_panelT.setRange({0, 0}, kk_offset);
      // continue update previous panels
      // Note: The tasks of the final huge TRSM of the HEGST step have been reshuffled to avoid extra
      // communication of the matrix L.
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
    }
    // TODO: With incomplete panel support this branch will disappears.
    else {
      if (kk_rank.row() == this_rank.row()) {
        hpx::shared_future<matrix::Tile<const T, Device::CPU>> l_kk;
        if (kk_rank.col() == this_rank.col()) {
          const GlobalTileIndex kk(k, k);
          l_kk = mat_l.read(kk);
          comm::scheduleSendBcast(executor_mpi, l_kk, mpi_row_task_chain());
        }
        else {
          l_kk = comm::scheduleRecvBcastAlloc<T, Device::CPU>(executor_mpi,
                                                              mat_l.tileSize(GlobalTileIndex{k, k}),
                                                              kk_rank.col(), mpi_row_task_chain());
        }
        for (SizeType j_local = 0; j_local < kk_offset.cols(); ++j_local) {
          const LocalTileIndex kj(kk_offset.rows(), j_local);
          trsmPanelUpdateTile(executor_hp, l_kk, mat_a(kj));
        }
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

    a_panel.setRangeStart(at_offset);

    hpx::shared_future<matrix::Tile<const T, Device::CPU>> a_diag;
    if (kk_rank.col() == this_rank.col()) {
      // Note:
      // [a,l]_panelT shrinked to a single tile for temporarly storing and communicating the diagonal
      // tile used for the column update
      a_panelT.setRange(kk_offset, at_offset);

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

    a_panelT.setRange(at_offset, distr.localNrTiles());

    // TODO skip last step tile
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

/// ---- ETI
#define DLAF_GENTOSTD_MC_ETI(KWORD, DATATYPE) \
  KWORD template struct GenToStd<Backend::MC, Device::CPU, DATATYPE>;

DLAF_GENTOSTD_MC_ETI(extern, float)
DLAF_GENTOSTD_MC_ETI(extern, double)
DLAF_GENTOSTD_MC_ETI(extern, std::complex<float>)
DLAF_GENTOSTD_MC_ETI(extern, std::complex<double>)
}
}
}
