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

#ifdef DLAF_WITH_GPU
#include <whip.hpp>
#endif

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/broadcast_panel.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/eigensolver/bt_reduction_to_band/api.h"
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
  static void copyAndSetHHUpperTiles(SizeType j_diag, const matrix::Tile<const T, Device::CPU>& src,
                                     const matrix::Tile<T, Device::CPU>& dst) {
    matrix::internal::copy_o(src, dst);
    lapack::laset(blas::Uplo::Upper, dst.size().rows(), dst.size().cols() - j_diag, T{0.}, T{1.},
                  dst.ptr({0, j_diag}), dst.ld());
  }
};

#ifdef DLAF_WITH_GPU
template <>
struct Helpers<Backend::GPU> {
  template <class T>
  static void copyAndSetHHUpperTiles(SizeType j_diag, const matrix::Tile<const T, Device::GPU>& src,
                                     const matrix::Tile<T, Device::GPU>& dst, whip::stream_t stream) {
    matrix::internal::copy_o(src, dst, stream);
    gpulapack::laset(blas::Uplo::Upper, dst.size().rows(), dst.size().cols() - j_diag, T{0.}, T{1.},
                     dst.ptr({0, j_diag}), dst.ld(), stream);
  }
};
#endif

template <Backend backend, typename SrcSender, typename DstSender>
void copyAndSetHHUpperTiles(SizeType j_diag, SrcSender&& src, DstSender&& dst) {
  namespace ex = pika::execution::experimental;
  using ElementType = dlaf::internal::SenderElementType<DstSender>;

  ex::start_detached(
      dlaf::internal::transform(dlaf::internal::Policy<backend>(pika::execution::thread_priority::high),
                                Helpers<backend>::template copyAndSetHHUpperTiles<ElementType>,
                                dlaf::internal::whenAllLift(j_diag, std::forward<SrcSender>(src),
                                                            std::forward<DstSender>(dst))));
}

template <Backend backend, class TSender, class SourcePanelSender, class PanelTileSender>
void trmmPanel(pika::execution::thread_priority priority, TSender&& t, SourcePanelSender&& v,
               PanelTileSender&& w) {
  using ElementType = dlaf::internal::SenderElementType<PanelTileSender>;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Upper, blas::Op::ConjTrans,
                                  blas::Diag::NonUnit, ElementType(1.0), std::forward<TSender>(t),
                                  std::forward<SourcePanelSender>(v), std::forward<PanelTileSender>(w)) |
      tile::trmm3(dlaf::internal::Policy<backend>(priority)));
}

template <Backend backend, class PanelTileSender, class MatrixTileSender, class ColPanelSender>
void gemmUpdateW2(pika::execution::thread_priority priority, PanelTileSender&& w, MatrixTileSender&& c,
                  ColPanelSender&& w2) {
  using ElementType = dlaf::internal::SenderElementType<PanelTileSender>;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Op::ConjTrans, blas::Op::NoTrans, ElementType(1.0),
                                  std::forward<PanelTileSender>(w), std::forward<MatrixTileSender>(c),
                                  ElementType(1.0), std::forward<ColPanelSender>(w2)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)));
}

template <Backend backend, class PanelTileSender, class ColPanelSender, class MatrixTileSender>
void gemmTrailingMatrix(pika::execution::thread_priority priority, PanelTileSender&& v,
                        ColPanelSender&& w2, MatrixTileSender&& c) {
  using ElementType = dlaf::internal::SenderElementType<PanelTileSender>;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, ElementType(-1.0),
                                  std::forward<PanelTileSender>(v), std::forward<ColPanelSender>(w2),
                                  ElementType(1.0), std::forward<MatrixTileSender>(c)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)));
}
}

// Implementation based on:
// G. H. Golub and C. F. Van Loan, Matrix Computations, chapter 5, The Johns Hopkins University Press

template <Backend backend, Device device, class T>
void BackTransformationReductionToBand<backend, device, T>::call(
    const SizeType b, Matrix<T, device>& mat_c, Matrix<const T, device>& mat_v,
    common::internal::vector<pika::shared_future<common::internal::vector<T>>> taus) {
  using namespace bt_red_band;
  using pika::execution::experimental::keep_future;
  auto hp = pika::execution::thread_priority::high;
  auto np = pika::execution::thread_priority::normal;

  const SizeType m = mat_c.nrTiles().rows();
  const SizeType n = mat_c.nrTiles().cols();
  const SizeType mb = mat_v.blockSize().rows();

  if (m <= 1 || n == 0)
    return;

  // Note: "-1" added to deal with size 1 reflector.
  const SizeType total_nr_reflector = mat_v.size().rows() - b - 1;

  if (total_nr_reflector <= 0)
    return;

  const auto dist_v = mat_v.distribution();
  const auto dist_c = mat_c.distribution();

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> panelsV(n_workspaces, dist_v);
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> panelsW(n_workspaces, dist_v);
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> panelsW2(n_workspaces, dist_c);

  dlaf::matrix::Distribution dist_t({mb, total_nr_reflector}, {mb, mb});
  matrix::Panel<Coord::Row, T, device> panelT(dist_t);

  const SizeType nr_reflector_blocks = dist_t.nrTiles().cols();

  for (SizeType k = nr_reflector_blocks - 1; k >= 0; --k) {
    bool is_last = (k == nr_reflector_blocks - 1);
    const SizeType nr_reflectors = dist_t.tileSize({0, k}).cols();

    const GlobalElementIndex v_offset(k * mb + b, k * mb);
    const GlobalElementIndex c_offset(k * mb + b, 0);

    const matrix::SubPanelView panel_view(dist_v, v_offset, nr_reflectors);
    const matrix::SubMatrixView mat_c_view(dist_c, c_offset);

    auto& panelV = panelsV.nextResource();
    auto& panelW = panelsW.nextResource();
    auto& panelW2 = panelsW2.nextResource();

    panelV.setRangeStart(v_offset);
    panelW.setRangeStart(v_offset);

    if (is_last) {
      panelT.setHeight(nr_reflectors);
      panelW2.setHeight(nr_reflectors);
      panelW.setWidth(nr_reflectors);
      panelV.setWidth(nr_reflectors);
    }

    for (const auto& i : panel_view.iteratorLocal()) {
      // Column index of the HH reflector which starts in the first row of this tile.
      const SizeType j_diag =
          std::max<SizeType>(0, i.row() * mat_v.blockSize().rows() - panel_view.offsetElement().row());

      if (j_diag < mb) {
        auto tile_v = splitTile(mat_v.read(i), panel_view(i));
        copyAndSetHHUpperTiles<backend>(j_diag, keep_future(tile_v), panelV.readwrite_sender(i));
      }
      else {
        panelV.setTile(i, mat_v.read(i));
      }
    }

    auto taus_panel = taus[k];
    const LocalTileIndex t_index{Coord::Col, k};
    dlaf::factorization::internal::computeTFactor<backend>(panelV, taus_panel, panelT(t_index));

    // W = V T
    auto tile_t = panelT.read_sender(t_index);
    for (const auto& idx : panelW.iteratorLocal()) {
      trmmPanel<backend>(np, tile_t, panelV.read_sender(idx), panelW.readwrite_sender(idx));
    }

    // W2 = W C
    matrix::util::set0<backend>(hp, panelW2);
    for (const auto& ij : mat_c_view.iteratorLocal()) {
      auto kj = LocalTileIndex{k, ij.col()};
      auto ik = LocalTileIndex{ij.row(), k};
      gemmUpdateW2<backend>(np, panelW.read_sender(ik),
                            keep_future(splitTile(mat_c.read(ij), mat_c_view(ij))),
                            panelW2.readwrite_sender(kj));
    }

    // Update trailing matrix: C = C - V W2
    for (const auto& ij : mat_c_view.iteratorLocal()) {
      auto ik = LocalTileIndex{ij.row(), k};
      auto kj = LocalTileIndex{k, ij.col()};
      gemmTrailingMatrix<backend>(np, panelV.read_sender(ik), panelW2.read_sender(kj),
                                  splitTile(mat_c(ij), mat_c_view(ij)));
    }

    panelV.reset();
    panelW.reset();
    panelW2.reset();
    panelT.reset();
  }
}

template <Backend B, Device D, class T>
void BackTransformationReductionToBand<B, D, T>::call(
    comm::CommunicatorGrid grid, Matrix<T, D>& mat_c, Matrix<const T, D>& mat_v,
    common::internal::vector<pika::shared_future<common::internal::vector<T>>> taus) {
  namespace ex = pika::execution::experimental;
  using namespace bt_red_band;
  using pika::execution::experimental::keep_future;
  auto hp = pika::execution::thread_priority::high;
  auto np = pika::execution::thread_priority::normal;

  // Set up MPI
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
  common::RoundRobin<matrix::Panel<Coord::Col, T, D>> panelsV(n_workspaces, dist_v);
  common::RoundRobin<matrix::Panel<Coord::Col, T, D>> panelsW(n_workspaces, dist_v);
  common::RoundRobin<matrix::Panel<Coord::Row, T, D>> panelsW2(n_workspaces, dist_c);

  dlaf::matrix::Distribution dist_t({mb, total_nr_reflector}, {mb, mb}, grid.size(), this_rank,
                                    dist_v.sourceRankIndex());
  matrix::Panel<Coord::Row, T, D> panelT(dist_t);

  const SizeType nr_reflector_blocks = dist_t.nrTiles().cols();

  for (SizeType k = nr_reflector_blocks - 1; k >= 0; --k) {
    const bool is_last = (k == nr_reflector_blocks - 1);
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

    const comm::IndexT_MPI k_rank_col = dist_v.template rankGlobalTile<Coord::Col>(k);

    if (this_rank.col() == k_rank_col) {
      for (SizeType i_local = dist_v.template nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
           i_local < dist_c.localNrTiles().rows(); ++i_local) {
        const SizeType i = dist_v.template globalTileFromLocalTile<Coord::Row>(i_local);
        const LocalTileIndex ik_panel(Coord::Row, i_local);
        if (i == v_start.row()) {
          pika::shared_future<matrix::Tile<const T, D>> tile_v = mat_v.read(GlobalTileIndex(i, k));
          if (is_last) {
            tile_v = splitTile(tile_v,
                               {{0, 0}, {dist_v.tileSize(GlobalTileIndex(i, k)).rows(), nr_reflectors}});
          }
          copyAndSetHHUpperTiles<B>(0, keep_future(tile_v), panelV.readwrite_sender(ik_panel));
        }
        else {
          panelV.setTile(ik_panel, mat_v.read(GlobalTileIndex(i, k)));
        }
      }

      const SizeType k_local = dist_t.template localTileFromGlobalTile<Coord::Col>(k);
      const LocalTileIndex t_index{Coord::Col, k_local};
      auto taus_panel = taus[k_local];

      using dlaf::factorization::internal::computeTFactor;
      computeTFactor<B>(panelV, taus_panel, panelT(t_index), mpi_col_task_chain);

      // WH = V T
      for (SizeType i_local = dist_v.template nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
           i_local < dist_v.localNrTiles().rows(); ++i_local) {
        const LocalTileIndex ik_panel{Coord::Row, i_local};
        trmmPanel<B>(np, panelT.read_sender(t_index), panelV.read_sender(ik_panel),
                     panelW.readwrite_sender(ik_panel));
      }
    }

    matrix::util::set0<B>(hp, panelW2);

    broadcast(k_rank_col, panelW, mpi_row_task_chain);

    // W2 = W C
    for (SizeType i_local = dist_c.template nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
         i_local < dist_c.localNrTiles().rows(); ++i_local) {
      for (SizeType j_local = 0; j_local < dist_c.localNrTiles().cols(); ++j_local) {
        const LocalTileIndex ij{i_local, j_local};
        gemmUpdateW2<B>(np, panelW.readwrite_sender(ij), mat_c.read_sender(ij),
                        panelW2.readwrite_sender(ij));
      }
    }

    for (const auto& kj_panel : panelW2.iteratorLocal())
      ex::start_detached(
          dlaf::comm::scheduleAllReduceInPlace(mpi_col_task_chain(), MPI_SUM,
                                               pika::execution::experimental::make_unique_any_sender(
                                                   panelW2.readwrite_sender(kj_panel))));

    broadcast(k_rank_col, panelV, mpi_row_task_chain);

    // C = C - V W2
    for (SizeType i_local = dist_c.template nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
         i_local < dist_c.localNrTiles().rows(); ++i_local) {
      for (SizeType j_local = 0; j_local < dist_c.localNrTiles().cols(); ++j_local) {
        const LocalTileIndex ij(i_local, j_local);
        gemmTrailingMatrix<B>(np, panelV.read_sender(ij), panelW2.read_sender(ij),
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
