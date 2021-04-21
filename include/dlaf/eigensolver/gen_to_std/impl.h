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

#include <hpx/local/execution.hpp>
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

template <Backend backend, class AKKSender, class LKKSender>
void hegstDiagTile(AKKSender&& a_kk, LKKSender&& l_kk) {
  dlaf::internal::whenAllLift(1, blas::Uplo::Lower, std::forward<AKKSender>(a_kk),
                              std::forward<LKKSender>(l_kk)) |
      dlaf::tile::hegst(dlaf::internal::Policy<backend>(hpx::threads::thread_priority::high)) |
      hpx::execution::experimental::detach();
}

template <Backend backend, class T, class LKKSender, class AIKSender>
void trsmPanelTile(LKKSender&& l_kk, AIKSender&& a_ik) {
  dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Lower, blas::Op::ConjTrans,
                              blas::Diag::NonUnit, T(1.0), std::forward<LKKSender>(l_kk),
                              std::forward<AIKSender>(a_ik)) |
      dlaf::tile::trsm(dlaf::internal::Policy<backend>(hpx::threads::thread_priority::high)) |
      hpx::execution::experimental::detach();
}

template <Backend backend, class T, class AKKSender, class LIKSender, class AIKSender>
void hemmPanelTile(AKKSender&& a_kk, LIKSender&& l_ik, AIKSender&& a_ik) {
  dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Lower, T(-0.5),
                              std::forward<AKKSender>(a_kk), std::forward<LIKSender>(l_ik), T(1.0),
                              std::forward<AIKSender>(a_ik)) |
      dlaf::tile::hemm(dlaf::internal::Policy<backend>(hpx::threads::thread_priority::high)) |
      hpx::execution::experimental::detach();
}

template <Backend backend, class T, class AJKSender, class LJKSender, class AKKSender>
void her2kTrailingDiagTile(hpx::threads::thread_priority priority, AJKSender&& a_jk, LJKSender&& l_jk,
                           AKKSender&& a_kk) {
  dlaf::internal::whenAllLift(blas::Uplo::Lower, blas::Op::NoTrans, T(-1.0),
                              std::forward<AJKSender>(a_jk), std::forward<LJKSender>(l_jk),
                              BaseType<T>(1.0), std::forward<AKKSender>(a_kk)) |
      dlaf::tile::her2k(dlaf::internal::Policy<backend>(priority)) |
      hpx::execution::experimental::detach();
}

template <Backend backend, class T, class MatIKSender, class MatJKSender, class AIJSender>
void gemmTrailingMatrixTile(hpx::threads::thread_priority priority, MatIKSender&& mat_ik,
                            MatJKSender&& mat_jk, AIJSender a_ij) {
  dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::ConjTrans, T(-1.0),
                              std::forward<MatIKSender>(mat_ik), std::forward<MatJKSender>(mat_jk),
                              T(1.0), std::forward<AIJSender>(a_ij)) |
      dlaf::tile::gemm(dlaf::internal::Policy<backend>(priority)) |
      hpx::execution::experimental::detach();
}

template <Backend backend, class T, class LJJSender, class AJKSender>
void trsmPanelUpdateTile(LJJSender&& l_jj, AJKSender a_jk) {
  dlaf::internal::whenAllLift(blas::Side::Left, blas::Uplo::Lower, blas::Op::NoTrans,
                              blas::Diag::NonUnit, T(1.0), std::forward<LJJSender>(l_jj),
                              std::forward<AJKSender>(a_jk)) |
      dlaf::tile::trsm(dlaf::internal::Policy<backend>(hpx::threads::thread_priority::high)) |
      hpx::execution::experimental::detach();
}

template <Backend backend, class T, class LIJSender, class AJKSender, class AIKSender>
void gemmPanelUpdateTile(LIJSender&& l_ij, AJKSender&& a_jk, AIKSender&& a_ik) {
  dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, T(-1.0),
                              std::forward<LIJSender>(l_ij), std::forward<AJKSender>(a_jk), T(1.0),
                              std::forward<AIKSender>(a_ik)) |
      dlaf::tile::gemm(dlaf::internal::Policy<backend>(hpx::threads::thread_priority::normal)) |
      hpx::execution::experimental::detach();
}

namespace gentostd_u {
template <class Executor, Device device, class T>
void hegstDiagTile(Executor&& executor_hp, hpx::future<matrix::Tile<T, device>> a_kk,
                   hpx::future<matrix::Tile<T, device>> u_kk) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::internal::hegst_o), 1, blas::Uplo::Upper,
                std::move(a_kk), std::move(u_kk));
}

template <class Executor, Device device, class T>
void trsmPanelTile(Executor&& executor_hp, hpx::shared_future<matrix::Tile<const T, device>> u_kk,
                   hpx::future<matrix::Tile<T, device>> a_ki) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::internal::trsm_o), blas::Side::Left,
                blas::Uplo::Upper, blas::Op::ConjTrans, blas::Diag::NonUnit, T(1.0), u_kk,
                std::move(a_ki));
}

template <class Executor, Device device, class T>
void hemmPanelTile(Executor&& executor_hp, hpx::shared_future<matrix::Tile<const T, device>> a_kk,
                   hpx::shared_future<matrix::Tile<const T, device>> u_ki,
                   hpx::future<matrix::Tile<T, device>> a_ki) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::internal::hemm_o), blas::Side::Left,
                blas::Uplo::Upper, T(-0.5), a_kk, u_ki, T(1.0), std::move(a_ki));
}

template <class Executor, Device device, class T>
void her2kTrailingDiagTile(Executor&& ex, hpx::shared_future<matrix::Tile<const T, device>> a_ki,
                           hpx::shared_future<matrix::Tile<const T, device>> u_ki,
                           hpx::future<matrix::Tile<T, device>> a_ii) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::internal::her2k_o), blas::Uplo::Upper,
                blas::Op::ConjTrans, T(-1.0), a_ki, u_ki, BaseType<T>(1.0), std::move(a_ii));
}

template <class Executor, Device device, class T>
void gemmTrailingMatrixTile(Executor&& ex, hpx::shared_future<matrix::Tile<const T, device>> mat_ki,
                            hpx::shared_future<matrix::Tile<const T, device>> mat_kj,
                            hpx::future<matrix::Tile<T, device>> a_ij) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::internal::gemm_o), blas::Op::ConjTrans,
                blas::Op::NoTrans, T(-1.0), mat_ki, mat_kj, T(1.0), std::move(a_ij));
}

template <class Executor, Device device, class T>
void trsmPanelUpdateTile(Executor&& executor_hp, hpx::shared_future<matrix::Tile<const T, device>> u_ii,
                         hpx::future<matrix::Tile<T, device>> a_ki) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::internal::trsm_o), blas::Side::Right,
                blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit, T(1.0), u_ii,
                std::move(a_ki));
}

template <class Executor, Device device, class T>
void gemmPanelUpdateTile(Executor&& ex, hpx::shared_future<matrix::Tile<const T, device>> a_ki,
                         hpx::shared_future<matrix::Tile<const T, device>> u_ij,
                         hpx::future<matrix::Tile<T, device>> a_kj) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::internal::gemm_o), blas::Op::NoTrans,
                blas::Op::NoTrans, T(-1.0), a_ki, u_ij, T(1.0), std::move(a_kj));
}
}

// Implementation based on LAPACK Algorithm for the transformation from generalized to standard
// eigenproblem (xHEGST)
template <Backend backend, Device device, class T>
void GenToStd<backend, device, T>::call_L(Matrix<T, device>& mat_a, Matrix<T, device>& mat_l) {
  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().cols();

  for (SizeType k = 0; k < nrtile; ++k) {
    const LocalTileIndex kk{k, k};

    // Direct transformation to standard eigenvalue problem of the diagonal tile
    hegstDiagTile<backend>(mat_a.readwrite_sender(kk), mat_l.readwrite_sender(kk));

    // If there is no trailing matrix
    if (k == nrtile - 1)
      continue;

    for (SizeType i = k + 1; i < nrtile; ++i) {
      const LocalTileIndex ik{i, k};
      trsmPanelTile<backend, T>(mat_l.read_sender(kk), mat_a.readwrite_sender(ik));
      hemmPanelTile<backend, T>(mat_a.read_sender(kk), mat_l.read_sender(ik), mat_a(ik));
    }

    for (SizeType j = k + 1; j < nrtile; ++j) {
      const LocalTileIndex jj{j, j};
      const LocalTileIndex jk{j, k};
      // first trailing panel gets high priority (look ahead).
      const auto trailing_matrix_priority =
          (j == k + 1) ? hpx::threads::thread_priority::high : hpx::threads::thread_priority::normal;

      her2kTrailingDiagTile<backend, T>(trailing_matrix_priority, mat_a.read_sender(jk),
                                        mat_l.read_sender(jk), mat_a.readwrite_sender(jj));

      for (SizeType i = j + 1; i < nrtile; ++i) {
        const LocalTileIndex ik{i, k};
        const LocalTileIndex ij{i, j};
        gemmTrailingMatrixTile<backend, T>(trailing_matrix_priority, mat_a.read_sender(ik),
                                           mat_l.read_sender(jk), mat_a.readwrite_sender(ij));
        gemmTrailingMatrixTile<backend, T>(trailing_matrix_priority, mat_l.read_sender(ik),
                                           mat_a.read_sender(jk), mat_a.readwrite_sender(ij));
      }
    }

    for (SizeType i = k + 1; i < nrtile; ++i) {
      const LocalTileIndex ik{i, k};
      hemmPanelTile<backend, T>(mat_a.read_sender(kk), mat_l.read_sender(ik), mat_a(ik));
    }

    for (SizeType j = k + 1; j < nrtile; ++j) {
      const LocalTileIndex jj{j, j};
      const LocalTileIndex jk{j, k};
      trsmPanelUpdateTile<backend, T>(mat_l.read_sender(jj), mat_a.readwrite_sender(jk));

      for (SizeType i = j + 1; i < nrtile; ++i) {
        const LocalTileIndex ij{i, j};
        const LocalTileIndex ik{i, k};
        gemmPanelUpdateTile<backend, T>(mat_l.read_sender(ij), mat_a.read_sender(jk),
                                        mat_a.readwrite_sender(ik));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void GenToStd<backend, device, T>::call_L(comm::CommunicatorGrid grid, Matrix<T, device>& mat_a,
                                          Matrix<T, device>& mat_l) {
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
    const GlobalTileIndex at{k + 1, k + 1};

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
    l_panel.setRangeStart(kk);
    l_panelT.setRangeStart(kk);
    a_panelT.setRange({0, 0}, kk);

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
    //       Therefore, all previous panels have to be updated at each step.
    if (kk_rank.row() == this_rank.row()) {
      for (SizeType j_local = 0; j_local < kk_offset.cols(); ++j_local) {
        const LocalTileIndex kk_panel(Coord::Row, kk_offset.rows());
        const LocalTileIndex kj_panelT{Coord::Col, j_local};
        const LocalTileIndex kj(kk_offset.rows(), j_local);

        trsmPanelUpdateTile<backend, T>(l_panel.read_sender(kk_panel), mat_a(kj));

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

          gemmPanelUpdateTile<backend, T>(l_panel.read_sender(ik_panel), a_panelT.read_sender(kj_panelT),
                                          mat_a.readwrite_sender(ij));
        }
      }
    }

    a_panelT.reset();

    // Direct transformation to standard eigenvalue problem of the diagonal tile
    if (kk_rank == this_rank)
      hegstDiagTile<backend>(mat_a.readwrite_sender(kk), mat_l.readwrite_sender(kk));

    // If there is no trailing matrix
    if (k == nrtile - 1)
      continue;

    const LocalTileIndex diag_wp_idx{0, kk_offset.cols()};

    a_panel.setRangeStart(at);

    hpx::shared_future<matrix::Tile<const T, device>> a_diag;
    if (kk_rank.col() == this_rank.col()) {
      // Note:
      // [a,l]_panelT shrinked to a single tile for temporarly storing and communicating the diagonal
      // tile used for the column update
      a_panelT.setRange(kk, at);

      if (kk_rank.row() == this_rank.row()) {
        a_panelT.setTile(diag_wp_idx, mat_a.read(kk));
      }
      broadcast(executor_mpi, kk_rank.row(), a_panelT, mpi_col_task_chain);

      // panel partial update
      for (SizeType i_local = at_offset.rows(); i_local < distr.localNrTiles().rows(); ++i_local) {
        const LocalTileIndex ik_panel(Coord::Row, i_local);
        const LocalTileIndex ik(i_local, distr.localTileFromGlobalTile<Coord::Col>(k));

        trsmPanelTile<backend, T>(l_panelT.read_sender(diag_wp_idx), mat_a.readwrite_sender(ik));
        hemmPanelTile<backend, T>(a_panelT.read_sender(diag_wp_idx), mat_l.read_sender(ik), mat_a(ik));

        // keep diagonal tile for later.
        a_diag = a_panelT.read(diag_wp_idx);

        a_panel.setTile(ik_panel, mat_a.read(ik));
      }

      // row panel has been used for temporary storage of diagonal panel for column update
      a_panelT.reset();
    }

    a_panelT.setRange(at, common::indexFromOrigin(distr.nrTiles()));

    broadcast(executor_mpi, kk_rank.col(), a_panel, a_panelT, mpi_row_task_chain, mpi_col_task_chain);

    // trailing matrix update
    for (SizeType j = k + 1; j < nrtile; ++j) {
      const auto owner = distr.rankGlobalTile({j, j});

      if (owner.col() != this_rank.col())
        continue;

      const auto j_local = distr.localTileFromGlobalTile<Coord::Col>(j);
      // first trailing panel gets high priority (look ahead).
      const auto trailing_matrix_priority =
          (j == k + 1) ? hpx::threads::thread_priority::high : hpx::threads::thread_priority::normal;
      if (this_rank.row() == owner.row()) {
        const auto i_local = distr.localTileFromGlobalTile<Coord::Row>(j);

        her2kTrailingDiagTile<backend, T>(trailing_matrix_priority,
                                          a_panel.read_sender({Coord::Row, i_local}),
                                          l_panel.read_sender({Coord::Row, i_local}),
                                          mat_a.readwrite_sender(LocalTileIndex{i_local, j_local}));
      }

      for (SizeType i = j + 1; i < nrtile; ++i) {
        const auto owner_row = distr.rankGlobalTile<Coord::Row>(i);

        if (owner_row != this_rank.row())
          continue;

        const auto i_local = distr.localTileFromGlobalTile<Coord::Row>(i);
        const LocalTileIndex ik_panel{Coord::Row, i_local};
        const LocalTileIndex kj_panelT{Coord::Col, j_local};
        const LocalTileIndex ij{i_local, j_local};

        gemmTrailingMatrixTile<backend, T>(hpx::threads::thread_priority::normal,
                                           a_panel.read_sender(ik_panel),
                                           l_panelT.read_sender(kj_panelT), mat_a.readwrite_sender(ij));
        gemmTrailingMatrixTile<backend, T>(hpx::threads::thread_priority::normal,
                                           l_panel.read_sender(ik_panel),
                                           a_panelT.read_sender(kj_panelT), mat_a.readwrite_sender(ij));
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

        hemmPanelTile<backend, T>(hpx::execution::experimental::keep_future(a_diag),
                                  mat_l.read_sender(ik), mat_a.readwrite_sender(ik));
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
    const GlobalTileIndex at{k + 1, k + 1};

    const LocalTileSize kk_offset{
        distr.nextLocalTileFromGlobalTile<Coord::Row>(k),
        distr.nextLocalTileFromGlobalTile<Coord::Col>(k),
    };

    const LocalTileSize at_offset{
        distr.nextLocalTileFromGlobalTile<Coord::Row>(k + 1),
        distr.nextLocalTileFromGlobalTile<Coord::Col>(k + 1),
    };

    auto& u_panel = u_panels.nextResource();
    auto& u_panelT = u_panelsT.nextResource();
    auto& a_panel = a_panels.nextResource();
    auto& a_panelT = a_panelsT.nextResource();
    u_panel.setRangeStart(kk);
    u_panelT.setRangeStart(kk);
    a_panelT.setRange({0, 0}, kk);

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

    // continue update previous panels
    // Note: The tasks of the final huge TRSM of the HEGST step have been reshuffled to avoid extra
    //       communication of the matrix U.
    //       During k-th iteration only the tasks involving the k-th panel of U are executed.
    //       Therefore, all previous panels have to be updated at each step.
    if (kk_rank.col() == this_rank.col()) {
      for (SizeType i_local = 0; i_local < kk_offset.rows(); ++i_local) {
        const LocalTileIndex kk_panel(Coord::Col, kk_offset.cols());
        const LocalTileIndex ki_panelT{Coord::Row, i_local};
        const LocalTileIndex ik(i_local, kk_offset.cols());

        trsmPanelUpdateTile(executor_hp, u_panel.read(kk_panel), mat_a(ik));

        a_panelT.setTile(ki_panelT, mat_a.read(ik));
      }
    }

    // No next rows update if last col.
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

    a_panel.setRangeStart(at);

    hpx::shared_future<matrix::Tile<const T, device>> a_diag;
    if (kk_rank.row() == this_rank.row()) {
      // Note:
      // [a,u]_panelT shrinked to a single tile for temporarly storing and communicating the diagonal
      // tile used for the row update
      a_panelT.setRange(kk, at);

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

      // col panel has been used for temporary storage of diagonal panel for row update
      a_panelT.reset();
    }

    a_panelT.setRange(at, common::indexFromOrigin(distr.nrTiles()));

    broadcast(executor_mpi, kk_rank.row(), a_panel, a_panelT, mpi_row_task_chain, mpi_col_task_chain);

    // trailing matrix update
    for (SizeType i = k + 1; i < nrtile; ++i) {
      const auto owner = distr.rankGlobalTile({i, i});

      if (owner.row() != this_rank.row())
        continue;

      const auto i_local = distr.localTileFromGlobalTile<Coord::Row>(i);
      // first trailing panel gets high priority (look ahead).
      auto& trailing_matrix_executor = (i == k + 1) ? executor_hp : executor_np;
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
