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

#include <pika/future.hpp>
#include <pika/thread.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/broadcast_panel.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/eigensolver/backtransformation/api.h"
#include "dlaf/executors.h"
#include "dlaf/factorization/qr.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

template <class T>
void copySingleTile(pika::shared_future<matrix::Tile<const T, Device::CPU>> in,
                    pika::future<matrix::Tile<T, Device::CPU>> out) {
  pika::dataflow(dlaf::getCopyExecutor<Device::CPU, Device::CPU>(),
                 matrix::unwrapExtendTiles(matrix::internal::copy_o), in, std::move(out));
}

template <class Executor, Device device, class T>
void trmmPanel(Executor&& ex, pika::shared_future<matrix::Tile<const T, device>> t,
               pika::future<matrix::Tile<T, device>> w) {
  pika::dataflow(ex, matrix::unwrapExtendTiles(tile::internal::trmm_o), blas::Side::Right,
                 blas::Uplo::Upper, blas::Op::ConjTrans, blas::Diag::NonUnit, T(1.0), t, std::move(w));
}

template <class Executor, Device device, class T>
void gemmUpdateW2(Executor&& ex, pika::future<matrix::Tile<T, device>> w,
                  pika::shared_future<matrix::Tile<const T, device>> c,
                  pika::future<matrix::Tile<T, device>> w2) {
  pika::dataflow(ex, matrix::unwrapExtendTiles(tile::internal::gemm_o), blas::Op::ConjTrans,
                 blas::Op::NoTrans, T(1.0), w, c, T(1.0), std::move(w2));
}

template <class Executor, Device device, class T>
void gemmTrailingMatrix(Executor&& ex, pika::shared_future<matrix::Tile<const T, device>> v,
                        pika::shared_future<matrix::Tile<const T, device>> w2,
                        pika::future<matrix::Tile<T, device>> c) {
  pika::dataflow(ex, matrix::unwrapExtendTiles(tile::internal::gemm_o), blas::Op::NoTrans,
                 blas::Op::NoTrans, T(-1.0), v, w2, T(1.0), std::move(c));
}

// Implementation based on:
// 1. Algorithm 6 "LAPACK Algorithm for the eigenvector back-transformation", page 15, PhD thesis
// "GPU Accelerated Implementations of a Generalized Eigenvalue Solver for Hermitian Matrices with
// Systematic Energy and Time to Solution Analysis" presented by Raffaele Solcà (2016)
// 2. G. H. Golub and C. F. Van Loan, Matrix Computations, chapter 5, The Johns Hopkins University Press
template <class T>
struct BackTransformation<Backend::MC, Device::CPU, T> {
  static void call_FC(Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v,
                      common::internal::vector<pika::shared_future<common::internal::vector<T>>> taus);
  static void call_FC(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_c,
                      Matrix<const T, Device::CPU>& mat_v,
                      common::internal::vector<pika::shared_future<common::internal::vector<T>>> taus);
};

template <class T>
void BackTransformation<Backend::MC, Device::CPU, T>::call_FC(
    Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v,
    common::internal::vector<pika::shared_future<common::internal::vector<T>>> taus) {
  auto executor_np = dlaf::getNpExecutor<Backend::MC>();

  const SizeType m = mat_c.nrTiles().rows();
  const SizeType n = mat_c.nrTiles().cols();
  const SizeType mb = mat_v.blockSize().rows();

  if (m <= 1 || n == 0)
    return;

  // Note: "-1" added to deal with size 1 reflector.
  const SizeType total_nr_reflector = mat_v.size().rows() - mb - 1;

  if (total_nr_reflector == 0)
    return;

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panelsV(n_workspaces,
                                                                        mat_v.distribution());
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panelsW(n_workspaces,
                                                                        mat_v.distribution());
  common::RoundRobin<matrix::Panel<Coord::Row, T, Device::CPU>> panelsW2(n_workspaces,
                                                                         mat_c.distribution());

  dlaf::matrix::Distribution dist_t({mb, total_nr_reflector}, {mb, mb});
  matrix::Panel<Coord::Row, T, Device::CPU> panelT(dist_t);

  const SizeType nr_reflector_blocks = dist_t.nrTiles().cols();

  for (SizeType k = nr_reflector_blocks - 1; k >= 0; --k) {
    bool is_last = (k == nr_reflector_blocks - 1);
    const GlobalTileIndex v_start{k + 1, k};

    auto& panelV = panelsV.nextResource();
    auto& panelW = panelsW.nextResource();
    auto& panelW2 = panelsW2.nextResource();

    panelV.setRangeStart(v_start);
    panelW.setRangeStart(v_start);

    const SizeType nr_reflectors = dist_t.tileSize({0, k}).cols();
    if (is_last) {
      panelT.setHeight(nr_reflectors);
      panelW2.setHeight(nr_reflectors);
      panelW.setWidth(nr_reflectors);
      panelV.setWidth(nr_reflectors);
    }

    for (SizeType i = k + 1; i < mat_v.nrTiles().rows(); ++i) {
      auto ik = LocalTileIndex{i, k};
      if (i == k + 1) {
        pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_v = mat_v.read(ik);
        if (is_last) {
          tile_v =
              splitTile(tile_v,
                        {{0, 0},
                         {mat_v.distribution().tileSize(GlobalTileIndex(i, k)).rows(), nr_reflectors}});
        }
        copySingleTile(tile_v, panelV(ik));
        pika::dataflow(pika::launch::sync, matrix::unwrapExtendTiles(tile::internal::laset_o),
                       lapack::MatrixType::Upper, T(0), T(1), panelV(ik));
      }
      else {
        panelV.setTile(ik, mat_v.read(ik));
      }
    }

    auto taus_panel = taus[k];
    const LocalTileIndex t_index{Coord::Col, k};
    dlaf::factorization::internal::computeTFactor<Backend::MC>(nr_reflectors, mat_v, v_start, taus_panel,
                                                               panelT(t_index));

    // W = V T
    pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_t = panelT.read(t_index);
    for (const auto& idx : panelW.iteratorLocal()) {
      copySingleTile(panelV.read(idx), panelW(idx));
      trmmPanel(executor_np, tile_t, panelW(idx));
    }

    // W2 = W C
    matrix::util::set0<Backend::MC>(pika::threads::thread_priority::high, panelW2);
    LocalTileIndex c_start{k + 1, 0};
    LocalTileIndex c_end{m, n};
    auto c_k = iterate_range2d(c_start, c_end);
    for (const auto& idx : c_k) {
      auto kj = LocalTileIndex{k, idx.col()};
      auto ik = LocalTileIndex{idx.row(), k};
      gemmUpdateW2(executor_np, panelW(ik), mat_c.read(idx), panelW2(kj));
    }

    // Update trailing matrix: C = C - V W2
    for (const auto& idx : c_k) {
      auto ik = LocalTileIndex{idx.row(), k};
      auto kj = LocalTileIndex{k, idx.col()};
      gemmTrailingMatrix(executor_np, panelV.read(ik), panelW2.read(kj), mat_c(idx));
    }

    panelV.reset();
    panelW.reset();
    panelW2.reset();
    panelT.reset();
  }
}

template <class T>
void BackTransformation<Backend::MC, Device::CPU, T>::call_FC(
    comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v,
    common::internal::vector<pika::shared_future<common::internal::vector<T>>> taus) {
  auto executor_np = dlaf::getNpExecutor<Backend::MC>();

  // Set up MPI
  auto executor_mpi = dlaf::getMPIExecutor<Backend::MC>();
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator().clone());
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator().clone());

  auto dist_v = mat_v.distribution();
  auto dist_c = mat_c.distribution();

  const SizeType m = mat_c.nrTiles().rows();
  const SizeType n = mat_c.nrTiles().cols();
  const SizeType mb = mat_v.blockSize().rows();

  const comm::Index2D this_rank = grid.rank();

  if (m <= 1 || n == 0)
    return;

  // Note: "-1" added to deal with size 1 reflector.
  const SizeType total_nr_reflector = mat_v.size().cols() - mb - 1;

  if (total_nr_reflector == 0)
    return;

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panelsV(n_workspaces, dist_v);
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panelsW(n_workspaces, dist_v);
  common::RoundRobin<matrix::Panel<Coord::Row, T, Device::CPU>> panelsW2(n_workspaces, dist_c);

  dlaf::matrix::Distribution dist_t({mb, total_nr_reflector}, {mb, mb}, grid.size(), this_rank,
                                    dist_v.sourceRankIndex());
  matrix::Panel<Coord::Row, T, Device::CPU> panelT(dist_t);

  const SizeType nr_reflector_blocks = dist_t.nrTiles().cols();

  for (SizeType k = nr_reflector_blocks - 1; k >= 0; --k) {
    bool is_last = (k == nr_reflector_blocks - 1);
    const GlobalTileIndex v_start{k + 1, k};

    auto& panelV = panelsV.nextResource();
    auto& panelW = panelsW.nextResource();
    auto& panelW2 = panelsW2.nextResource();

    panelV.setRangeStart(v_start);
    panelW.setRangeStart(v_start);
    panelT.setRange(GlobalTileIndex(Coord::Col, k), GlobalTileIndex(Coord::Col, k + 1));

    const SizeType nr_reflectors = dist_t.tileSize({0, k}).cols();
    if (is_last) {
      panelT.setHeight(nr_reflectors);
      panelW2.setHeight(nr_reflectors);
      panelW.setWidth(nr_reflectors);
      panelV.setWidth(nr_reflectors);
    }

    auto k_rank_col = dist_v.template rankGlobalTile<Coord::Col>(k);

    if (this_rank.col() == k_rank_col) {
      for (SizeType i_local = dist_v.template nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
           i_local < dist_c.localNrTiles().rows(); ++i_local) {
        auto i = dist_v.template globalTileFromLocalTile<Coord::Row>(i_local);
        auto ik_panel = LocalTileIndex{Coord::Row, i_local};
        if (i == v_start.row()) {
          pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_v =
              mat_v.read(GlobalTileIndex(i, k));
          if (is_last) {
            tile_v = splitTile(tile_v,
                               {{0, 0}, {dist_v.tileSize(GlobalTileIndex(i, k)).rows(), nr_reflectors}});
          }
          copySingleTile(tile_v, panelV(ik_panel));
          pika::dataflow(pika::launch::sync, matrix::unwrapExtendTiles(tile::internal::laset_o),
                         lapack::MatrixType::Upper, T(0), T(1), panelV(ik_panel));
        }
        else {
          panelV.setTile(ik_panel, mat_v.read(GlobalTileIndex(i, k)));
        }
      }

      auto k_local = dist_t.template localTileFromGlobalTile<Coord::Col>(k);
      const LocalTileIndex t_index{Coord::Col, k_local};
      auto taus_panel = taus[k_local];
      dlaf::factorization::internal::computeTFactor<Backend::MC>(nr_reflectors, mat_v, v_start,
                                                                 taus_panel, panelT(t_index),
                                                                 mpi_col_task_chain);

      for (SizeType i_local = dist_v.template nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
           i_local < dist_v.localNrTiles().rows(); ++i_local) {
        // WH = V T
        const LocalTileIndex ik_panel{Coord::Row, i_local};
        copySingleTile(panelV.read(ik_panel), panelW(ik_panel));
        trmmPanel(executor_np, panelT.read(t_index), panelW(ik_panel));
      }
    }

    matrix::util::set0<Backend::MC>(pika::threads::thread_priority::high, panelW2);

    broadcast(executor_mpi, k_rank_col, panelW, mpi_row_task_chain);

    for (SizeType i_local = dist_c.template nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
         i_local < dist_c.localNrTiles().rows(); ++i_local) {
      const LocalTileIndex ik_panel{Coord::Row, i_local};
      for (SizeType j_local = 0; j_local < dist_c.localNrTiles().cols(); ++j_local) {
        // W2 = W C
        const LocalTileIndex kj_panel{Coord::Col, j_local};
        const LocalTileIndex ij{i_local, j_local};
        gemmUpdateW2(executor_np, panelW(ik_panel), mat_c.read(ij), panelW2(kj_panel));
      }
    }

    for (const auto& kj_panel : panelW2.iteratorLocal())
      scheduleAllReduceInPlace(executor_mpi, mpi_col_task_chain(), MPI_SUM, panelW2(kj_panel));

    broadcast(executor_mpi, k_rank_col, panelV, mpi_row_task_chain);

    for (SizeType i_local = dist_c.template nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
         i_local < dist_c.localNrTiles().rows(); ++i_local) {
      const LocalTileIndex ik_panel{Coord::Row, i_local};
      for (SizeType j_local = 0; j_local < dist_c.localNrTiles().cols(); ++j_local) {
        // C = C - V W2
        const LocalTileIndex kj_panel{Coord::Col, j_local};
        const LocalTileIndex ij(i_local, j_local);
        gemmTrailingMatrix(executor_np, panelV.read(ik_panel), panelW2.read(kj_panel), mat_c(ij));
      }
    }

    panelV.reset();
    panelW.reset();
    panelW2.reset();
    panelT.reset();
  }
}

/// ---- ETI
#define DLAF_EIGENSOLVER_BACKTRANSFORMATION_MC_ETI(KWORD, DATATYPE) \
  KWORD template struct BackTransformation<Backend::MC, Device::CPU, DATATYPE>;

DLAF_EIGENSOLVER_BACKTRANSFORMATION_MC_ETI(extern, float)
DLAF_EIGENSOLVER_BACKTRANSFORMATION_MC_ETI(extern, double)
DLAF_EIGENSOLVER_BACKTRANSFORMATION_MC_ETI(extern, std::complex<float>)
DLAF_EIGENSOLVER_BACKTRANSFORMATION_MC_ETI(extern, std::complex<double>)

}
}
}
