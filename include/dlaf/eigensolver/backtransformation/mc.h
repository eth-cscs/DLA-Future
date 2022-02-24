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
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/matrix/views.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

template <typename InSender, typename OutSender>
void copySingleTile(InSender&& in, OutSender&& out) {
  pika::execution::experimental::when_all(std::forward<InSender>(in), std::forward<OutSender>(out)) |
      matrix::copy(
          dlaf::internal::Policy<dlaf::matrix::internal::CopyBackend_v<Device::CPU, Device::CPU>>()) |
      pika::execution::experimental::start_detached();
}

template <typename T, typename TSender, typename WSender>
void trmmPanel(pika::threads::thread_priority priority, TSender&& t, WSender&& w) {
  dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Upper, blas::Op::ConjTrans,
                              blas::Diag::NonUnit, T(1.0), std::forward<TSender>(t),
                              std::forward<WSender>(w)) |
      tile::trmm(dlaf::internal::Policy<Backend::MC>(priority)) |
      pika::execution::experimental::start_detached();
}

template <typename T, typename WSender, typename CSender, typename W2Sender>
void gemmUpdateW2(pika::threads::thread_priority priority, WSender&& w, CSender&& c, W2Sender&& w2) {
  dlaf::internal::whenAllLift(blas::Op::ConjTrans, blas::Op::NoTrans, T(1.0), std::forward<WSender>(w),
                              std::forward<CSender>(c), T(1.0), std::forward<W2Sender>(w2)) |
      tile::gemm(dlaf::internal::Policy<Backend::MC>(priority)) |
      pika::execution::experimental::start_detached();
}

template <typename T, typename VSender, typename W2Sender, typename CSender>
void gemmTrailingMatrix(pika::threads::thread_priority priority, VSender&& v, W2Sender&& w2,
                        CSender&& c) {
  dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, T(-1.0), std::forward<VSender>(v),
                              std::forward<W2Sender>(w2), T(1.0), std::forward<CSender>(c)) |
      tile::gemm(dlaf::internal::Policy<Backend::MC>(priority)) |
      pika::execution::experimental::start_detached();
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
  using pika::execution::experimental::keep_future;
  using pika::threads::thread_priority;

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
        copySingleTile(keep_future(tile_v), panelV.readwrite_sender(ik));
        // TODO: How important is launch::sync here?
        dlaf::internal::whenAllLift(lapack::MatrixType::Upper, T(0), T(1), panelV.readwrite_sender(ik)) |
            tile::laset(dlaf::internal::Policy<Backend::MC>()) |
            pika::execution::experimental::start_detached();
      }
      else {
        panelV.setTile(ik, mat_v.read(ik));
      }
    }

    const GlobalElementIndex v_offset(v_start.row() * mb, v_start.col() * mb);
    const matrix::SubPanelView panel_view(mat_v.distribution(), v_offset, nr_reflectors);

    auto taus_panel = taus[k];
    const LocalTileIndex t_index{Coord::Col, k};
    dlaf::factorization::internal::computeTFactor<Backend::MC>(nr_reflectors, mat_v, panel_view,
                                                               taus_panel, panelT(t_index));

    // W = V T
    pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_t = panelT.read(t_index);
    for (const auto& idx : panelW.iteratorLocal()) {
      copySingleTile(panelV.read_sender(idx), panelW.readwrite_sender(idx));
      trmmPanel<T>(thread_priority::normal, keep_future(tile_t), panelW.readwrite_sender(idx));
    }

    // W2 = W C
    matrix::util::set0<Backend::MC>(pika::threads::thread_priority::high, panelW2);
    LocalTileIndex c_start{k + 1, 0};
    LocalTileIndex c_end{m, n};
    auto c_k = iterate_range2d(c_start, c_end);
    for (const auto& idx : c_k) {
      auto kj = LocalTileIndex{k, idx.col()};
      auto ik = LocalTileIndex{idx.row(), k};
      gemmUpdateW2<T>(thread_priority::normal, panelW.readwrite_sender(ik), mat_c.read_sender(idx),
                      panelW2.readwrite_sender(kj));
    }

    // Update trailing matrix: C = C - V W2
    for (const auto& idx : c_k) {
      auto ik = LocalTileIndex{idx.row(), k};
      auto kj = LocalTileIndex{k, idx.col()};
      gemmTrailingMatrix<T>(thread_priority::normal, panelV.read_sender(ik), panelW2.read_sender(kj),
                            mat_c.readwrite_sender(idx));
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
  using pika::execution::experimental::keep_future;
  using pika::threads::thread_priority;

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
          copySingleTile(keep_future(tile_v), panelV.readwrite_sender(ik_panel));
          // TODO: How important is launch::sync here?
          dlaf::internal::whenAllLift(lapack::MatrixType::Upper, T(0), T(1),
                                      panelV.readwrite_sender(ik_panel)) |
              tile::laset(dlaf::internal::Policy<Backend::MC>()) |
              pika::execution::experimental::start_detached();
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
        copySingleTile(panelV.read_sender(ik_panel), panelW.readwrite_sender(ik_panel));
        trmmPanel<T>(thread_priority::normal, panelT.read_sender(t_index),
                     panelW.readwrite_sender(ik_panel));
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
        gemmUpdateW2<T>(thread_priority::normal, panelW.readwrite_sender(ik_panel),
                        mat_c.read_sender(ij), panelW2.readwrite_sender(kj_panel));
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
        gemmTrailingMatrix<T>(thread_priority::normal, panelV.read_sender(ik_panel),
                              panelW2.read_sender(kj_panel), mat_c.readwrite_sender(ij));
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
