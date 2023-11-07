//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <pika/execution.hpp>
#include <pika/thread.hpp>

#ifdef DLAF_WITH_GPU
#include <whip.hpp>
#endif

#include <dlaf/blas/tile.h>
#include <dlaf/common/index2d.h>
#include <dlaf/common/pipeline.h>
#include <dlaf/common/round_robin.h>
#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/communication/broadcast_panel.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/kernels.h>
#include <dlaf/eigensolver/bt_reduction_to_band/api.h>
#include <dlaf/factorization/qr.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/copy_tile.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/layout_info.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/panel.h>
#include <dlaf/matrix/views.h>
#include <dlaf/util_matrix.h>

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
    common::internal::SingleThreadedBlasScope single;
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

  ex::start_detached(dlaf::internal::transform(
      dlaf::internal::Policy<backend>(pika::execution::thread_priority::high, pika::execution::thread_stacksize::nostack),
      Helpers<backend>::template copyAndSetHHUpperTiles<ElementType>,
      dlaf::internal::whenAllLift(j_diag, std::forward<SrcSender>(src), std::forward<DstSender>(dst))));
}

template <Backend backend, class TSender, class SourcePanelSender, class PanelTileSender>
void trmmPanel(pika::execution::thread_priority priority, TSender&& t, SourcePanelSender&& v,
               PanelTileSender&& w) {
  using ElementType = dlaf::internal::SenderElementType<PanelTileSender>;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Upper, blas::Op::ConjTrans,
                                  blas::Diag::NonUnit, ElementType(1.0), std::forward<TSender>(t),
                                  std::forward<SourcePanelSender>(v), std::forward<PanelTileSender>(w)) |
      tile::trmm3(dlaf::internal::Policy<backend>(priority, pika::execution::thread_stacksize::nostack)));
}

template <Backend backend, class PanelTileSender, class MatrixTileSender, class ColPanelSender>
void gemmUpdateW2(pika::execution::thread_priority priority, PanelTileSender&& w, MatrixTileSender&& c,
                  ColPanelSender&& w2) {
  using ElementType = dlaf::internal::SenderElementType<PanelTileSender>;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Op::ConjTrans, blas::Op::NoTrans, ElementType(1.0),
                                  std::forward<PanelTileSender>(w), std::forward<MatrixTileSender>(c),
                                  ElementType(1.0), std::forward<ColPanelSender>(w2)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority, pika::execution::thread_stacksize::nostack)));
}

template <Backend backend, class PanelTileSender, class ColPanelSender, class MatrixTileSender>
void gemmTrailingMatrix(pika::execution::thread_priority priority, PanelTileSender&& v,
                        ColPanelSender&& w2, MatrixTileSender&& c) {
  using ElementType = dlaf::internal::SenderElementType<PanelTileSender>;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, ElementType(-1.0),
                                  std::forward<PanelTileSender>(v), std::forward<ColPanelSender>(w2),
                                  ElementType(1.0), std::forward<MatrixTileSender>(c)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority, pika::execution::thread_stacksize::nostack)));
}
}

// Implementation based on:
// G. H. Golub and C. F. Van Loan, Matrix Computations, chapter 5, The Johns Hopkins University Press

template <Backend backend, Device device, class T>
void BackTransformationReductionToBand<backend, device, T>::call(
    const SizeType b, Matrix<T, device>& mat_c, Matrix<const T, device>& mat_v,
    Matrix<const T, Device::CPU>& mat_taus) {
  using namespace bt_red_band;

  auto hp = pika::execution::thread_priority::high;
  auto np = pika::execution::thread_priority::normal;

  const SizeType m = mat_c.nrTiles().rows();
  const SizeType n = mat_c.nrTiles().cols();
  const SizeType mb = mat_v.blockSize().rows();

  if (m == 0 || n == 0)
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

      if (j_diag < nr_reflectors) {
        auto tile_v = splitTile(mat_v.read(i), panel_view(i));
        copyAndSetHHUpperTiles<backend>(j_diag, std::move(tile_v), panelV.readwrite(i));
      }
      else if (j_diag < mb) {
        panelV.setTile(i, splitTile(mat_v.read(i), panel_view(i)));
      }
      else {
        panelV.setTile(i, mat_v.read(i));
      }
    }

    const LocalTileIndex taus_index{Coord::Row, k};
    const LocalTileIndex t_index{Coord::Col, k};
    dlaf::factorization::internal::computeTFactor<backend>(panelV, mat_taus.read(taus_index),
                                                           panelT.readwrite(t_index));

    // W = V T
    auto tile_t = panelT.read(t_index);
    for (const auto& idx : panelW.iteratorLocal()) {
      trmmPanel<backend>(np, tile_t, panelV.read(idx), panelW.readwrite(idx));
    }

    // W2 = W C
    matrix::util::set0<backend>(hp, panelW2);
    for (const auto& ij : mat_c_view.iteratorLocal()) {
      gemmUpdateW2<backend>(np, panelW.read(ij), splitTile(mat_c.read(ij), mat_c_view(ij)),
                            panelW2.readwrite(ij));
    }

    // Update trailing matrix: C = C - V W2
    for (const auto& ij : mat_c_view.iteratorLocal()) {
      gemmTrailingMatrix<backend>(np, panelV.read(ij), panelW2.read(ij),
                                  splitTile(mat_c.readwrite(ij), mat_c_view(ij)));
    }

    panelV.reset();
    panelW.reset();
    panelW2.reset();
    panelT.reset();
  }
}

template <Backend B, Device D, class T>
void BackTransformationReductionToBand<B, D, T>::call(comm::CommunicatorGrid grid, const SizeType b,
                                                      Matrix<T, D>& mat_c, Matrix<const T, D>& mat_v,
                                                      Matrix<const T, Device::CPU>& mat_taus) {
  namespace ex = pika::execution::experimental;
  using namespace bt_red_band;

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

  if (m == 0 || n == 0)
    return;

  // Note: "-1" added to deal with size 1 reflector.
  const SizeType total_nr_reflector = mat_v.size().cols() - b - 1;

  if (total_nr_reflector <= 0)
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
    panelT.setRange(GlobalTileIndex(Coord::Col, k), GlobalTileIndex(Coord::Col, k + 1));

    if (is_last) {
      panelT.setHeight(nr_reflectors);
      panelW2.setHeight(nr_reflectors);
      panelW.setWidth(nr_reflectors);
      panelV.setWidth(nr_reflectors);
    }

    const comm::IndexT_MPI k_rank_col = dist_v.template rankGlobalTile<Coord::Col>(k);

    if (this_rank.col() == k_rank_col) {
      for (const auto& ik : panel_view.iteratorLocal()) {
        const SizeType i_row_g = dist_v.template globalTileFromLocalTile<Coord::Row>(ik.row());

        // Column index of the HH reflector which starts in the first row of this tile.
        const SizeType j_diag =
            std::max<SizeType>(0, i_row_g * mat_v.blockSize().rows() - panel_view.offsetElement().row());

        if (j_diag < nr_reflectors) {
          auto tile_v = splitTile(mat_v.read(ik), panel_view(ik));
          copyAndSetHHUpperTiles<B>(j_diag, tile_v, panelV.readwrite(ik));
        }
        else if (j_diag < mb) {
          panelV.setTile(ik, splitTile(mat_v.read(ik), panel_view(ik)));
        }
        else {
          panelV.setTile(ik, mat_v.read(ik));
        }
      }

      const GlobalTileIndex taus_index{Coord::Row, k};
      const SizeType k_local = dist_t.template localTileFromGlobalTile<Coord::Col>(k);
      const LocalTileIndex t_index{Coord::Col, k_local};
      dlaf::factorization::internal::computeTFactor<B>(panelV, mat_taus.read(taus_index),
                                                       panelT.readwrite(t_index), mpi_col_task_chain);

      // WH = V T
      for (const auto& idx : panel_view.iteratorLocal()) {
        trmmPanel<B>(np, panelT.read(t_index), panelV.read(idx), panelW.readwrite(idx));
      }
    }

    matrix::util::set0<B>(hp, panelW2);

    broadcast(k_rank_col, panelW, mpi_row_task_chain);

    // W2 = W C
    for (const auto& ij : mat_c_view.iteratorLocal()) {
      gemmUpdateW2<B>(np, panelW.readwrite(ij), splitTile(mat_c.read(ij), mat_c_view(ij)),
                      panelW2.readwrite(ij));
    }

    for (const auto& kj_panel : panelW2.iteratorLocal())
      ex::start_detached(dlaf::comm::scheduleAllReduceInPlace(mpi_col_task_chain(), MPI_SUM,
                                                              panelW2.readwrite(kj_panel)));

    broadcast(k_rank_col, panelV, mpi_row_task_chain);

    // C = C - V W2
    for (const auto& ij : mat_c_view.iteratorLocal()) {
      gemmTrailingMatrix<B>(np, panelV.read(ij), panelW2.read(ij),
                            splitTile(mat_c.readwrite(ij), mat_c_view(ij)));
    }

    panelV.reset();
    panelW.reset();
    panelW2.reset();
    panelT.reset();
  }
}
}
