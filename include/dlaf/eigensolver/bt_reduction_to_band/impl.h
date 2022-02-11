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
#include "dlaf/eigensolver/bt_reduction_to_band/api.h"
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

namespace dlaf::eigensolver::internal {

namespace bt_red_band {

template <Backend B>
struct Helpers;

template <>
struct Helpers<Backend::MC> {
  template <class T>
  static void copyAndSetHHUpperTile(const matrix::Tile<const T, Device::CPU>& src,
                                    matrix::Tile<T, Device::CPU>&& dst) {
    matrix::internal::copy_o(src, dst);
    tile::internal::laset_o(blas::Uplo::Upper, T{0.}, T{1.}, dst);
  }
};

template <Backend backend, typename SrcSender, typename DstSender>
void copyAndSetHHUpperTile(SrcSender&& src, DstSender&& dst) {
  namespace ex = pika::execution::experimental;
  using ElementType = dlaf::internal::SenderElementType<DstSender>;

  dlaf::internal::transform(dlaf::internal::Policy<backend>(pika::threads::thread_priority::high),
                            Helpers<backend>::template copyAndSetHHUpperTile<ElementType>,
                            ex::when_all(std::forward<SrcSender>(src), std::forward<DstSender>(dst))) |
      ex::start_detached();
}

template <Backend backend, class TSender, class SourcePanelSender, class PanelTileSender>
void trmmPanel(pika::threads::thread_priority priority, TSender&& t, SourcePanelSender&& v,
               PanelTileSender&& w) {
  using ElementType = dlaf::internal::SenderElementType<PanelTileSender>;

  dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Upper, blas::Op::ConjTrans,
                              blas::Diag::NonUnit, ElementType(1.0), std::forward<TSender>(t),
                              std::forward<SourcePanelSender>(v), std::forward<PanelTileSender>(w)) |
      tile::trmm3(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}

template <Backend backend, class PanelTileSender, class MatrixTileSender, class ColPanelSender>
void gemmUpdateW2(pika::threads::thread_priority priority, PanelTileSender&& w, MatrixTileSender&& c,
                  ColPanelSender&& w2) {
  using ElementType = dlaf::internal::SenderElementType<PanelTileSender>;

  dlaf::internal::whenAllLift(blas::Op::ConjTrans, blas::Op::NoTrans, ElementType(1.0),
                              std::forward<PanelTileSender>(w), std::forward<MatrixTileSender>(c),
                              ElementType(1.0), std::forward<ColPanelSender>(w2)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}

template <Backend backend, class PanelTileSender, class ColPanelSender, class MatrixTileSender>
void gemmTrailingMatrix(pika::threads::thread_priority priority, PanelTileSender&& v,
                        ColPanelSender&& w2, MatrixTileSender&& c) {
  using ElementType = dlaf::internal::SenderElementType<PanelTileSender>;

  dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, ElementType(-1.0),
                              std::forward<PanelTileSender>(v), std::forward<ColPanelSender>(w2),
                              ElementType(1.0), std::forward<MatrixTileSender>(c)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}
}

// Implementation based on:
// G. H. Golub and C. F. Van Loan, Matrix Computations, chapter 5, The Johns Hopkins University Press

template <Backend backend, Device device, class T>
void BackTransformationReductionToBand<backend, device, T>::call(
    Matrix<T, device>& mat_c, Matrix<const T, device>& mat_v,
    common::internal::vector<pika::shared_future<common::internal::vector<T>>> taus) {
  using namespace bt_red_band;
  using pika::execution::experimental::keep_future;
  auto hp = pika::threads::thread_priority::high;
  auto np = pika::threads::thread_priority::normal;

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
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> panelsV(n_workspaces, mat_v.distribution());
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> panelsW(n_workspaces, mat_v.distribution());
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> panelsW2(n_workspaces, mat_c.distribution());

  dlaf::matrix::Distribution dist_t({mb, total_nr_reflector}, {mb, mb});
  matrix::Panel<Coord::Row, T, device> panelT(dist_t);

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
        auto tile_v = mat_v.read(ik);
        if (is_last) {
          tile_v =
              splitTile(tile_v,
                        {{0, 0},
                         {mat_v.distribution().tileSize(GlobalTileIndex(i, k)).rows(), nr_reflectors}});
        }
        copyAndSetHHUpperTile<backend>(keep_future(tile_v), panelV.readwrite_sender(ik));
      }
      else {
        panelV.setTile(ik, mat_v.read(ik));
      }
    }

    const GlobalElementIndex v_offset(v_start.row() * mb, v_start.col() * mb);
    const matrix::SubPanelView panel_view(mat_v.distribution(), v_offset, nr_reflectors);

    auto taus_panel = taus[k];
    const LocalTileIndex t_index{Coord::Col, k};
    dlaf::factorization::internal::computeTFactor<backend>(nr_reflectors, mat_v, panel_view, taus_panel,
                                                           panelT(t_index));

    // W = V T
    auto tile_t = panelT.read_sender(t_index);
    for (const auto& idx : panelW.iteratorLocal()) {
      trmmPanel<backend>(np, tile_t, panelV.read_sender(idx), panelW.readwrite_sender(idx));
    }

    // W2 = W C
    matrix::util::set0<backend>(hp, panelW2);
    LocalTileIndex c_start{k + 1, 0};
    LocalTileIndex c_end{m, n};
    auto c_k = iterate_range2d(c_start, c_end);
    for (const auto& idx : c_k) {
      auto kj = LocalTileIndex{k, idx.col()};
      auto ik = LocalTileIndex{idx.row(), k};
      gemmUpdateW2<backend>(np, panelW(ik), mat_c.read_sender(idx), panelW2.readwrite_sender(kj));
    }

    // Update trailing matrix: C = C - V W2
    for (const auto& idx : c_k) {
      auto ik = LocalTileIndex{idx.row(), k};
      auto kj = LocalTileIndex{k, idx.col()};
      gemmTrailingMatrix<backend>(np, panelV.read_sender(ik), panelW2.read_sender(kj),
                                  mat_c.readwrite_sender(idx));
    }

    panelV.reset();
    panelW.reset();
    panelW2.reset();
    panelT.reset();
  }
}

template <Backend backend, Device device, class T>
void BackTransformationReductionToBand<backend, device, T>::call(
    comm::CommunicatorGrid grid, Matrix<T, device>& mat_c, Matrix<const T, device>& mat_v,
    common::internal::vector<pika::shared_future<common::internal::vector<T>>> taus) {
  using namespace bt_red_band;
  using pika::execution::experimental::keep_future;
  auto hp = pika::threads::thread_priority::high;
  auto np = pika::threads::thread_priority::normal;

  if constexpr (backend != Backend::MC) {
    DLAF_STATIC_UNIMPLEMENTED(T);
  }

  // Set up MPI
  auto executor_mpi = dlaf::getMPIExecutor<backend>();
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
          copyAndSetHHUpperTile<backend>(keep_future(tile_v), panelV.readwrite_sender(ik_panel));
        }
        else {
          panelV.setTile(ik_panel, mat_v.read(GlobalTileIndex(i, k)));
        }
      }

      auto k_local = dist_t.template localTileFromGlobalTile<Coord::Col>(k);
      const LocalTileIndex t_index{Coord::Col, k_local};
      auto taus_panel = taus[k_local];
      dlaf::factorization::internal::computeTFactor<backend>(nr_reflectors, mat_v, v_start, taus_panel,
                                                             panelT(t_index), mpi_col_task_chain);

      for (SizeType i_local = dist_v.template nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
           i_local < dist_v.localNrTiles().rows(); ++i_local) {
        // WH = V T
        const LocalTileIndex ik_panel{Coord::Row, i_local};
        trmmPanel<backend>(np, panelT.read_sender(t_index), panelV.read_sender(ik_panel),
                           panelW.readwrite_sender(ik_panel));
      }
    }

    matrix::util::set0<backend>(hp, panelW2);

    broadcast(executor_mpi, k_rank_col, panelW, mpi_row_task_chain);

    for (SizeType i_local = dist_c.template nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
         i_local < dist_c.localNrTiles().rows(); ++i_local) {
      const LocalTileIndex ik_panel{Coord::Row, i_local};
      for (SizeType j_local = 0; j_local < dist_c.localNrTiles().cols(); ++j_local) {
        // W2 = W C
        const LocalTileIndex kj_panel{Coord::Col, j_local};
        const LocalTileIndex ij{i_local, j_local};
        gemmUpdateW2<backend>(np, panelW(ik_panel), mat_c.read_sender(ij),
                              panelW2.readwrite_sender(kj_panel));
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
        gemmTrailingMatrix<backend>(np, panelV.read_sender(ik_panel), panelW2.read_sender(kj_panel),
                                    mat_c.readwrite_sender(ij));
      }
    }

    panelV.reset();
    panelW.reset();
    panelW2.reset();
    panelT.reset();
  }
}
}
