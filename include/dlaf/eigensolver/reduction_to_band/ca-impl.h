//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <pika/execution.hpp>

#include <dlaf/communication/broadcast_panel.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/kernels/all_reduce.h>
#include <dlaf/communication/kernels/broadcast.h>
#include <dlaf/eigensolver/reduction_to_band/api.h>
#include <dlaf/eigensolver/reduction_to_band/common.h>
#include <dlaf/factorization/qr.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/copy_tile.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/panel.h>
#include <dlaf/schedulers.h>
#include <dlaf/sender/transform.h>
#include <dlaf/sender/transform_mpi.h>

//
#include <dlaf/matrix/print_numpy.h>

namespace dlaf::eigensolver::internal {

namespace ca_red2band {
template <Backend B, Device D, class T>
void hemm(comm::Index2D rank_qr, matrix::Panel<Coord::Col, T, D>& W1,
          matrix::Panel<Coord::Row, T, D, matrix::StoreTransposed::Yes>& W1T,
          const matrix::SubMatrixView& at_view, matrix::Matrix<const T, D>& A,
          matrix::Panel<Coord::Col, const T, D>& W0,
          matrix::Panel<Coord::Row, T, D, matrix::StoreTransposed::Yes>& W0T,
          comm::CommunicatorPipeline<comm::CommunicatorType::Row>& mpi_row_chain,
          comm::CommunicatorPipeline<comm::CommunicatorType::Col>& mpi_col_chain) {
  namespace ex = pika::execution::experimental;

  using red2band::hemmDiag;
  using red2band::hemmOffDiag;

  using pika::execution::thread_priority;

  const auto dist = A.distribution();
  const auto rank = dist.rankIndex();

  // Note:
  // They have to be set to zero, because all tiles are going to be reduced, and some tiles may not get
  // "initialized" during computation, so they should not contribute with any spurious value to final result.
  matrix::util::set0<B>(thread_priority::high, W1);
  matrix::util::set0<B>(thread_priority::high, W1T);

  const LocalTileIndex at_offset = at_view.begin();

  for (SizeType i_lc = at_offset.row(); i_lc < dist.local_nr_tiles().rows(); ++i_lc) {
    // Note:
    // diagonal included: get where the first upper tile is in local coordinates
    const SizeType i = dist.template global_tile_from_local_tile<Coord::Row>(i_lc);
    const auto j_end_lc = dist.template next_local_tile_from_global_tile<Coord::Col>(i + 1);

    for (SizeType j_lc = j_end_lc - 1; j_lc >= at_offset.col(); --j_lc) {
      const LocalTileIndex ij_lc{i_lc, j_lc};
      const GlobalTileIndex ij = dist.global_tile_index(ij_lc);

      const bool is_diagonal_tile = (ij.row() == ij.col());

      auto getSubA = [&A, &at_view, ij_lc]() { return splitTile(A.read(ij_lc), at_view(ij_lc)); };

      if (is_diagonal_tile) {
        const comm::IndexT_MPI id_qr_R = dist.template rank_global_tile<Coord::Row>(ij.col());

        // Note:
        // Use W0 just if the tile belongs to the current local transformation.
        if (id_qr_R != rank_qr.row())
          continue;

        hemmDiag<B>(thread_priority::high, getSubA(), W0.read(ij_lc), W1.readwrite(ij_lc));
      }
      else {
        const GlobalTileIndex ijL = ij;
        const comm::IndexT_MPI id_qr_lower_R = dist.template rank_global_tile<Coord::Row>(ijL.col());
        if (id_qr_lower_R == rank_qr.row()) {
          // Note:
          // Since it is not a diagonal tile, otherwise it would have been managed in the previous
          // branch, the second operand might not be available in W but it is accessible through the
          // support panel W1T.
          // However, since we are still computing the "straight" part, the result can be stored
          // in the "local" panel W1.
          hemmOffDiag<B>(thread_priority::high, blas::Op::NoTrans, getSubA(), W0T.read(ij_lc),
                         W1.readwrite(ij_lc));
        }

        const GlobalTileIndex ijU = transposed(ij);
        const comm::IndexT_MPI id_qr_upper_R = dist.template rank_global_tile<Coord::Row>(ijU.col());
        if (id_qr_upper_R == rank_qr.row()) {
          // Note:
          // Here we are considering the hermitian part of A, so pretend to deal with transposed coordinate.
          // Check if the result still belongs to the same rank, otherwise store it in the support panel.
          const comm::IndexT_MPI owner_row = dist.template rank_global_tile<Coord::Row>(ijU.row());
          const SizeType iU_lc = dist.template local_tile_from_global_tile<Coord::Row>(ij.col());
          const LocalTileIndex i_w1_lc(iU_lc, 0);
          const LocalTileIndex i_w1t_lc(0, ij_lc.col());
          auto tile_w1 = (rank.row() == owner_row) ? W1.readwrite(i_w1_lc) : W1T.readwrite(i_w1t_lc);

          hemmOffDiag<B>(thread_priority::high, blas::Op::ConjTrans, getSubA(), W0.read(ij_lc),
                         std::move(tile_w1));
        }
      }
    }
  }

  // Note:
  // At this point, partial results of W1 are available in the panels, and they have to be reduced,
  // both row-wise and col-wise. The final W1 result will be available just on Ai panel column.

  // Note:
  // The first step in reducing partial results distributed over W1 and W1T, it is to reduce the row
  // panel W1T col-wise, by collecting all W1T results on the rank which can "mirror" the result on its
  // rows (i.e. diagonal). So, for each tile of the row panel, select who is the "diagonal" rank that can
  // mirror and reduce on it.
  if (mpi_col_chain.size() > 1) {
    for (const auto& i_wt_lc : W1T.iteratorLocal()) {
      const auto i_diag = dist.template global_tile_from_local_tile<Coord::Col>(i_wt_lc.col());
      const auto rank_owner_row = dist.template rank_global_tile<Coord::Row>(i_diag);

      if (rank_owner_row == rank.row()) {
        // Note:
        // Since it is the owner, it has to perform the "mirroring" of the results from columns to
        // rows.
        // Moreover, it reduces in place because the owner of the diagonal stores the partial result
        // directly in W1 (without using W1T)
        const auto i_w1_lc = dist.template local_tile_from_global_tile<Coord::Row>(i_diag);
        ex::start_detached(comm::schedule_reduce_recv_in_place(mpi_col_chain.exclusive(), MPI_SUM,
                                                               W1.readwrite({i_w1_lc, 0})));
      }
      else {
        ex::start_detached(comm::schedule_reduce_send(mpi_col_chain.exclusive(), rank_owner_row, MPI_SUM,
                                                      W1T.read(i_wt_lc)));
      }
    }
  }

  // Note:
  // At this point partial results are all collected in X (Xt has been embedded in previous step),
  // so the last step needed is to reduce these last partial results in the final results.
  if (mpi_row_chain.size() > 1) {
    for (const auto& i_w1_lc : W1.iteratorLocal()) {
      if (rank_qr.col() == rank.col())
        ex::start_detached(comm::schedule_reduce_recv_in_place(mpi_row_chain.exclusive(), MPI_SUM,
                                                               W1.readwrite(i_w1_lc)));
      else
        ex::start_detached(comm::schedule_reduce_send(mpi_row_chain.exclusive(), rank_qr.col(), MPI_SUM,
                                                      W1.read(i_w1_lc)));
    }
  }
}

template <Backend B, Device D, class T>
void her2kUpdateTrailingMatrix(comm::Index2D rank_qr, const matrix::SubMatrixView& at_view,
                               matrix::Matrix<T, D>& a, matrix::Panel<Coord::Col, const T, D>& W3,
                               matrix::Panel<Coord::Col, const T, D>& V) {
  static_assert(std::is_signed_v<BaseType<T>>, "alpha in computations requires to be -1");

  using pika::execution::thread_priority;
  using red2band::her2kDiag;
  using red2band::her2kOffDiag;

  const auto dist = a.distribution();
  const comm::Index2D rank = dist.rank_index();

  const LocalTileIndex at_offset = at_view.begin();

  if (rank_qr.row() != rank.row())
    return;

  for (SizeType i_lc = at_offset.row(); i_lc < dist.local_nr_tiles().rows(); ++i_lc) {
    // Note:
    // diagonal included: get where the first upper tile is in local coordinates
    const SizeType i = dist.template global_tile_from_local_tile<Coord::Row>(i_lc);
    const auto j_end_lc = dist.template next_local_tile_from_global_tile<Coord::Col>(i + 1);

    for (SizeType j_lc = j_end_lc - 1; j_lc >= at_offset.col(); --j_lc) {
      const LocalTileIndex ij_lc{i_lc, j_lc};
      const GlobalTileIndex ij = dist.global_tile_index(ij_lc);

      const comm::IndexT_MPI id_qr_L = dist.template rank_global_tile<Coord::Row>(ij.row());
      const comm::IndexT_MPI id_qr_R = dist.template rank_global_tile<Coord::Row>(ij.col());

      // Note: this computation applies just to tiles where transformation applies both from L and R
      if (id_qr_L != id_qr_R)
        continue;

      const bool is_diagonal_tile = (ij.row() == ij.col());

      auto getSubA = [&a, &at_view, ij_lc]() { return splitTile(a.readwrite(ij_lc), at_view(ij_lc)); };

      // The first column of the trailing matrix (except for the very first global tile) has to be
      // updated first, in order to unlock the next iteration as soon as possible.
      const auto priority = (j_lc == at_offset.col()) ? thread_priority::high : thread_priority::normal;

      if (is_diagonal_tile) {
        her2kDiag<B>(priority, V.read(ij_lc), W3.read(ij_lc), getSubA());
      }
      else {
        // TODO check and document why tranposed operand can be accessed with same index locally
        // A -= W3 . V*
        her2kOffDiag<B>(priority, W3.read(ij_lc), V.read(ij_lc), getSubA());
        // A -= V . W3*
        her2kOffDiag<B>(priority, V.read(ij_lc), W3.read(ij_lc), getSubA());
      }
    }
  }
}

template <Backend B, Device D, class T>
void hemm2nd(comm::IndexT_MPI rank_panel, matrix::Panel<Coord::Col, T, D>& W1,
             matrix::Panel<Coord::Row, T, D, matrix::StoreTransposed::Yes>& W1T,
             const matrix::SubMatrixView& at_view, const SizeType j_end, matrix::Matrix<const T, D>& A,
             matrix::Panel<Coord::Col, const T, D>& W0,
             matrix::Panel<Coord::Row, T, D, matrix::StoreTransposed::Yes>& W0T,
             comm::CommunicatorPipeline<comm::CommunicatorType::Row>& mpi_row_chain,
             comm::CommunicatorPipeline<comm::CommunicatorType::Col>& mpi_col_chain) {
  namespace ex = pika::execution::experimental;

  using red2band::hemmDiag;
  using red2band::hemmOffDiag;

  using pika::execution::thread_priority;

  const auto dist = A.distribution();
  const auto rank = dist.rankIndex();

  // Note:
  // They have to be set to zero, because all tiles are going to be reduced, and some tiles may not get
  // "initialized" during computation, so they should not contribute with any spurious value to final result.
  matrix::util::set0<B>(thread_priority::high, W1);
  matrix::util::set0<B>(thread_priority::high, W1T);

  const LocalTileIndex at_offset = at_view.begin();

  const SizeType jR_end_lc = dist.template next_local_tile_from_global_tile<Coord::Col>(j_end);
  for (SizeType i_lc = at_offset.row(); i_lc < dist.localNrTiles().rows(); ++i_lc) {
    const auto j_end_lc =
        std::min(jR_end_lc, dist.template next_local_tile_from_global_tile<Coord::Col>(
                                dist.template global_tile_from_local_tile<Coord::Row>(i_lc) + 1));
    for (SizeType j_lc = at_offset.col(); j_lc < j_end_lc; ++j_lc) {
      const LocalTileIndex ij_lc(i_lc, j_lc);
      const GlobalTileIndex ij = dist.global_tile_index(ij_lc);
      std::cout << "hemm2nd @ " << ij << "\n";

      // skip upper
      if (ij.row() < ij.col()) {
        std::cout << "hemm2nd-skipping " << ij << "\n";
        continue;
      }

      const bool is_diag = (ij.row() == ij.col());

      if (is_diag) {
        std::cout << "hemm2nd-diag " << ij << "\n";
        hemmDiag<B>(thread_priority::high, A.read(ij_lc), W0.read(ij_lc), W1.readwrite(ij_lc));
      }
      else {
        std::cout << "hemm2nd-odL " << ij << "\n";
        // Lower
        hemmOffDiag<B>(thread_priority::high, blas::Op::NoTrans, A.read(ij_lc), W0T.read(ij_lc),
                       W1.readwrite(ij_lc));

        // Upper
        const GlobalTileIndex ijU = transposed(ij);

        // Note: if it is out of the "sub-matrix"
        if (ijU.col() >= j_end)
          continue;

        std::cout << "hemm2nd-odU " << ij << "\n";

        const comm::IndexT_MPI owner_row = dist.template rank_global_tile<Coord::Row>(ijU.row());
        const SizeType iU_lc = dist.template local_tile_from_global_tile<Coord::Row>(ij.col());
        const LocalTileIndex i_w1_lc(iU_lc, 0);
        const LocalTileIndex i_w1t_lc(0, ij_lc.col());
        auto tile_w1 = (rank.row() == owner_row) ? W1.readwrite(i_w1_lc) : W1T.readwrite(i_w1t_lc);

        hemmOffDiag<B>(thread_priority::high, blas::Op::ConjTrans, A.read(ij_lc), W0.read(ij_lc),
                       std::move(tile_w1));
      }
    }
  }

  // Note:
  // At this point, partial results of W1 are available in the panels, and they have to be reduced,
  // both row-wise and col-wise. The final W1 result will be available just on Ai panel column.

  // Note:
  // The first step in reducing partial results distributed over W1 and W1T, it is to reduce the row
  // panel W1T col-wise, by collecting all W1T results on the rank which can "mirror" the result on its
  // rows (i.e. diagonal). So, for each tile of the row panel, select who is the "diagonal" rank that can
  // mirror and reduce on it.
  if (mpi_col_chain.size() > 1) {
    for (const auto& i_wt_lc : W1T.iteratorLocal()) {
      const auto i_diag = dist.template global_tile_from_local_tile<Coord::Col>(i_wt_lc.col());
      const auto rank_owner_row = dist.template rank_global_tile<Coord::Row>(i_diag);

      if (rank_owner_row == rank.row()) {
        // Note:
        // Since it is the owner, it has to perform the "mirroring" of the results from columns to
        // rows.
        // Moreover, it reduces in place because the owner of the diagonal stores the partial result
        // directly in W1 (without using W1T)
        const auto i_w1_lc = dist.template local_tile_from_global_tile<Coord::Row>(i_diag);
        ex::start_detached(comm::schedule_reduce_recv_in_place(mpi_col_chain.exclusive(), MPI_SUM,
                                                               W1.readwrite({i_w1_lc, 0})));
      }
      else {
        ex::start_detached(comm::schedule_reduce_send(mpi_col_chain.exclusive(), rank_owner_row, MPI_SUM,
                                                      W1T.read(i_wt_lc)));
      }
    }
  }

  // Note:
  // At this point partial results are all collected in X (Xt has been embedded in previous step),
  // so the last step needed is to reduce these last partial results in the final results.
  if (mpi_row_chain.size() > 1) {
    for (const auto& i_w1_lc : W1.iteratorLocal()) {
      if (rank_panel == rank.col())
        ex::start_detached(comm::schedule_reduce_recv_in_place(mpi_row_chain.exclusive(), MPI_SUM,
                                                               W1.readwrite(i_w1_lc)));
      else
        ex::start_detached(comm::schedule_reduce_send(mpi_row_chain.exclusive(), rank_panel, MPI_SUM,
                                                      W1.read(i_w1_lc)));
    }
  }
}

template <Backend B, Device D, class T>
void her2k_2nd(const SizeType j_end, const matrix::SubMatrixView& at_view, matrix::Matrix<T, D>& a,
               matrix::Panel<Coord::Col, const T, D>& W1,
               matrix::Panel<Coord::Row, T, D, matrix::StoreTransposed::Yes>& W1T,
               matrix::Panel<Coord::Col, const T, D>& V,
               matrix::Panel<Coord::Row, T, D, matrix::StoreTransposed::Yes>& VT) {
  static_assert(std::is_signed_v<BaseType<T>>, "alpha in computations requires to be -1");

  using pika::execution::thread_priority;
  using red2band::her2kDiag;
  using red2band::her2kOffDiag;

  const auto dist = a.distribution();

  const LocalTileIndex at_offset = at_view.begin();

  const SizeType jR_end_lc = dist.template next_local_tile_from_global_tile<Coord::Col>(j_end);
  for (SizeType i_lc = at_offset.row(); i_lc < dist.localNrTiles().rows(); ++i_lc) {
    const auto j_end_lc =
        std::min(jR_end_lc, dist.template next_local_tile_from_global_tile<Coord::Col>(
                                dist.template global_tile_from_local_tile<Coord::Row>(i_lc) + 1));
    for (SizeType j_lc = at_offset.col(); j_lc < j_end_lc; ++j_lc) {
      const LocalTileIndex ij_local{i_lc, j_lc};
      const GlobalTileIndex ij = dist.globalTileIndex(ij_local);

      const bool is_diagonal_tile = (ij.row() == ij.col());

      auto getSubA = [&a, &at_view, ij_local]() {
        return splitTile(a.readwrite(ij_local), at_view(ij_local));
      };

      // The first column of the trailing matrix (except for the very first global tile) has to be
      // updated first, in order to unlock the next iteration as soon as possible.
      const auto priority = (j_lc == at_offset.col()) ? thread_priority::high : thread_priority::normal;

      if (is_diagonal_tile) {
        her2kDiag<B>(priority, V.read(ij_local), W1.read(ij_local), getSubA());
      }
      else {
        // A -= X . V*
        her2kOffDiag<B>(priority, W1.read(ij_local), VT.read(ij_local), getSubA());

        // A -= V . X*
        her2kOffDiag<B>(priority, V.read(ij_local), W1T.read(ij_local), getSubA());
      }
    }
  }
}
}

// Distributed implementation of reduction to band
template <Backend B, Device D, class T>
CARed2BandResult<T, D> CAReductionToBand<B, D, T>::call(comm::CommunicatorGrid& grid,
                                                        Matrix<T, D>& mat_a, const SizeType band_size) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  using common::RoundRobin;
  using matrix::Panel;
  using matrix::StoreTransposed;

  const auto& dist = mat_a.distribution();
  const comm::Index2D rank = dist.rank_index();

  const SizeType nrefls = std::max<SizeType>(0, dist.size().cols() - band_size - 1);

  // Note:
  // Reflector of size = 1 is not considered whatever T is (i.e. neither real
  // nor complex)
  DLAF_ASSERT(dist.block_size().cols() % band_size == 0, dist.block_size().cols(), band_size);

  // Note:
  // row-vector that is distributed over columns, but replicated over rows.
  // for historical reason it is stored and accessed as a column-vector.
  DLAF_ASSERT(dist.block_size().cols() % band_size == 0, dist.block_size().cols(), band_size);
  const matrix::Distribution dist_taus(GlobalElementSize(nrefls, 1),
                                       TileElementSize(dist.block_size().cols(), 1),
                                       comm::Size2D(dist.grid_size().cols(), 1),
                                       comm::Index2D(rank.col(), 0),
                                       comm::Index2D(dist.source_rank_index().col(), 0));
  Matrix<T, Device::CPU> mat_taus_1st(dist_taus);
  Matrix<T, Device::CPU> mat_taus_2nd(dist_taus);

  // Note:
  // row-panel distributed over columns, but replicated over rows
  const matrix::Distribution dist_hh_2nd(GlobalElementSize(dist.block_size().rows(), dist.size().cols()),
                                         dist.block_size(), comm::Size2D(1, dist.grid_size().cols()),
                                         comm::Index2D(0, rank.col()),
                                         comm::Index2D(0, dist.source_rank_index().col()));
  Matrix<T, D> mat_hh_2nd(dist_hh_2nd);

  if (nrefls == 0)
    return {std::move(mat_taus_1st), std::move(mat_taus_2nd), std::move(mat_hh_2nd)};

  auto mpi_col_chain = grid.col_communicator_pipeline();
  auto mpi_row_chain = grid.row_communicator_pipeline();

  constexpr std::size_t n_workspaces = 2;

  // TODO HEADS workspace
  // - column vector
  // - has to be fully local
  // - no more than grid_size.rows() tiles (1 tile per rank in the column)
  // - we use panel just because it offers the ability to shrink width/height
  const matrix::Distribution dist_heads(
      LocalElementSize(dist.grid_size().rows() * dist.block_size().rows(), dist.block_size().cols()),
      dist.block_size());

  RoundRobin<Panel<Coord::Col, T, D>> panels_heads(n_workspaces, dist_heads);

  // update trailing matrix workspaces
  RoundRobin<Panel<Coord::Col, T, D>> panels_v(n_workspaces, dist);
  RoundRobin<Panel<Coord::Row, T, D, StoreTransposed::Yes>> panels_vt(n_workspaces, dist);

  RoundRobin<Panel<Coord::Col, T, D>> panels_w0(n_workspaces, dist);
  RoundRobin<Panel<Coord::Row, T, D, StoreTransposed::Yes>> panels_w0t(n_workspaces, dist);

  RoundRobin<Panel<Coord::Col, T, D>> panels_w1(n_workspaces, dist);
  RoundRobin<Panel<Coord::Row, T, D, StoreTransposed::Yes>> panels_w1t(n_workspaces, dist);

  RoundRobin<Panel<Coord::Col, T, D>> panels_w3(n_workspaces, dist);

  DLAF_ASSERT(mat_a.block_size().cols() == band_size, mat_a.block_size().cols(), band_size);
  const SizeType ntiles = (nrefls - 1) / band_size + 1;

  const bool is_full_band = (band_size == dist.blockSize().cols());

  for (SizeType j = 0; j < ntiles; ++j) {
    const SizeType i = j + 1;

    const SizeType nrefls_step = dist_taus.tile_size_of({j, 0}).rows();

    // panel
    const GlobalTileIndex panel_offset(i, j);
    const GlobalElementIndex panel_offset_el(panel_offset.row() * band_size,
                                             panel_offset.col() * band_size);
    matrix::SubPanelView panel_view(dist, panel_offset_el, band_size);

    const comm::IndexT_MPI rank_panel(dist.template rank_global_tile<Coord::Col>(panel_offset.col()));

    const SizeType n_qr_heads =
        std::min<SizeType>(panel_view.offset().row() + grid.size().rows(), dist.nr_tiles().rows()) -
        panel_view.offset().row();

    // trailing
    const GlobalTileIndex at_offset(i, j + 1);
    const GlobalElementIndex at_offset_el(at_offset.row() * band_size, at_offset.col() * band_size);
    const LocalTileIndex at_offset_lc(
        dist.template next_local_tile_from_global_tile<Coord::Row>(at_offset.row()),
        dist.template next_local_tile_from_global_tile<Coord::Col>(at_offset.col()));
    matrix::SubMatrixView at_view(dist, at_offset_el);

    // PANEL: just ranks in the current column
    // QR local (HH reflectors stored in-place)
    if (rank_panel == rank.col())
      red2band::local::computePanelReflectors(mat_a, mat_taus_1st, panel_offset.col(), panel_view);

    // TRAILING 1st pass
    if (at_offset_el.isIn(mat_a.size())) {
      auto& ws_V = panels_v.nextResource();

      ws_V.setRangeStart(at_offset);
      ws_V.setWidth(nrefls_step);

      const LocalTileIndex zero_lc(0, 0);
      matrix::Matrix<T, D> ws_T({nrefls_step, nrefls_step}, dist.block_size());

      if (rank_panel == rank.col()) {
        using factorization::internal::computeTFactor;
        using red2band::local::setupReflectorPanelV;

        const bool has_head = !panel_view.iteratorLocal().empty();
        setupReflectorPanelV<B, D, T>(has_head, panel_view, nrefls_step, ws_V, mat_a, !is_full_band);

        const GlobalTileIndex j_tau(j, 0);
        computeTFactor<B>(ws_V, mat_taus_1st.read(j_tau), ws_T.readwrite(zero_lc));
      }

      auto& ws_VT = panels_vt.nextResource();
      ws_VT.setRangeStart(at_offset);
      ws_VT.setHeight(nrefls_step);

      comm::broadcast(rank_panel, ws_V, ws_VT, mpi_row_chain, mpi_col_chain);

      // W = V T
      auto& ws_W0 = panels_w0.nextResource();
      auto& ws_W0T = panels_w0t.nextResource();

      ws_W0.setRangeStart(at_offset);
      ws_W0T.setRangeStart(at_offset);

      ws_W0.setWidth(nrefls_step);
      ws_W0T.setHeight(nrefls_step);

      if (rank_panel == rank.col())
        red2band::local::trmmComputeW<B, D>(ws_W0, ws_V, ws_T.read(zero_lc));

      comm::broadcast(rank_panel, ws_W0, ws_W0T, mpi_row_chain, mpi_col_chain);

      // Note: apply local transformations, one after the other
      auto& ws_W1 = panels_w1.nextResource();
      auto& ws_W1T = panels_w1t.nextResource();
      matrix::Matrix<T, D> ws_W2 = std::move(ws_T);

      for (int idx_qr_head = 0; idx_qr_head < n_qr_heads; ++idx_qr_head) {
        const SizeType head_qr = at_view.offset().row() + idx_qr_head;
        const comm::Index2D rank_qr(dist.template rank_global_tile<Coord::Row>(head_qr), rank_panel);

        ws_W1.setRangeStart(at_offset);
        ws_W1.setWidth(nrefls_step);

        ws_W1T.setRangeStart(at_offset);
        ws_W1T.setHeight(nrefls_step);

        // W1 = A W0
        // Note:
        // it will fill up all W1 (distributed) and it will read just part of A, i.e. columns where this
        // local transformation should be applied from the right.
        using ca_red2band::hemm;
        hemm<B>(rank_qr, ws_W1, ws_W1T, at_view, mat_a, ws_W0, ws_W0T, mpi_row_chain, mpi_col_chain);

        // Note:
        // W1T has been used as support panel, so reset it again.
        ws_W1T.reset();
        ws_W1T.setRangeStart(at_offset);
        ws_W1T.setHeight(nrefls_step);

        comm::broadcast(rank_panel, ws_W1, ws_W1T, mpi_row_chain, mpi_col_chain);

        // LR
        // A -= V W1* + W1 V* - V W0* W1 V*
        if (rank.row() == rank_qr.row()) {
          // W2 = W0.T W1
          red2band::local::gemmComputeW2<B, D>(ws_W2, ws_W0, ws_W1);

          // Note:
          // Next steps for L and R need W1, so we create a copy that we are going to update for this step.
          auto& ws_W3 = panels_w3.nextResource();
          ws_W3.setRangeStart(at_offset);
          ws_W3.setWidth(nrefls_step);

          for (const auto& idx : ws_W1.iteratorLocal())
            ex::start_detached(ex::when_all(ws_W1.read(idx), ws_W3.readwrite(idx)) |
                               matrix::copy(di::Policy<Backend::MC>{}));

          // W1 -= 0.5 V W2
          red2band::local::gemmUpdateX<B, D>(ws_W3, ws_W2, ws_V);
          // A -= W1 V.T + V W1.T
          ca_red2band::her2kUpdateTrailingMatrix<B>(rank_qr, at_view, mat_a, ws_W3, ws_V);

          ws_W3.reset();
        }

        // R (exclusively)
        // A -= W1 V*
        // Note: all rows, but just the columns that are in the local transformation rank
        for (SizeType j_lc = at_offset_lc.col(); j_lc < dist.local_nr_tiles().cols(); ++j_lc) {
          const SizeType j = dist.template global_tile_from_local_tile<Coord::Col>(j_lc);
          const comm::IndexT_MPI id_qr_R = dist.template rank_global_tile<Coord::Row>(j);

          if (rank_qr.row() != id_qr_R)
            continue;

          auto&& tile_vt = ws_VT.read({0, j_lc});

          for (SizeType i_lc = at_offset_lc.row(); i_lc < dist.local_nr_tiles().rows(); ++i_lc) {
            const LocalTileIndex ij_lc(i_lc, j_lc);
            const GlobalTileIndex ij = dist.global_tile_index(ij_lc);

            // TODO just lower part of trailing matrix
            if (ij.row() < ij.col())
              continue;

            const comm::IndexT_MPI id_qr_L = dist.template rank_global_tile<Coord::Row>(ij.row());

            // Note: exclusively from R, if it is an LR tile, it is computed elsewhere
            if (id_qr_L == id_qr_R)
              continue;

            ex::start_detached(di::whenAllLift(blas::Op::NoTrans, blas::Op::ConjTrans, T(-1),
                                               ws_W1.read(ij_lc), tile_vt, T(1),
                                               mat_a.readwrite(ij_lc)) |
                               tile::gemm(di::Policy<B>()));
          }
        }

        // L (exclusively)
        // A -= V W1*
        // Note: all cols, but just the rows of current transformation
        if (rank_qr.row() == rank.row()) {
          for (SizeType i_lc = at_offset_lc.row(); i_lc < dist.local_nr_tiles().rows(); ++i_lc) {
            const comm::IndexT_MPI id_qr_L = rank_qr.row();

            for (SizeType j_lc = at_offset_lc.col(); j_lc < dist.local_nr_tiles().cols(); ++j_lc) {
              const LocalTileIndex ij_lc(i_lc, j_lc);
              const GlobalTileIndex ij = dist.global_tile_index(ij_lc);

              // TODO just lower part of trailing matrix
              if (ij.row() < ij.col())
                continue;

              const comm::IndexT_MPI id_qr_R = dist.template rank_global_tile<Coord::Row>(ij.col());

              // Note: exclusively from L, if it is an LR tile, it is computed elsewhere
              if (id_qr_L == id_qr_R)
                continue;

              ex::start_detached(di::whenAllLift(blas::Op::NoTrans, blas::Op::ConjTrans, T(-1),
                                                 ws_V.read(ij_lc), ws_W1T.read(ij_lc), T(1),
                                                 mat_a.readwrite(ij_lc)) |
                                 tile::gemm(di::Policy<B>()));
            }
          }
        }

        ws_W1T.reset();
        ws_W1.reset();
      }

      ws_W0T.reset();
      ws_W0.reset();
      ws_VT.reset();
      ws_V.reset();
    }

    // ===== 2nd pass

    // PANEL: just ranks in the current column
    // QR local with just heads (HH reflectors have to be computed elsewhere, 1st-pass in-place)
    auto&& panel_heads = panels_heads.nextResource();
    panel_heads.setRangeEnd({n_qr_heads, 0});
    panel_heads.setWidth(nrefls_step);

    const matrix::Distribution dist_heads_current(LocalElementSize(n_qr_heads * dist.block_size().rows(),
                                                                   dist.block_size().cols()),
                                                  dist.block_size());
    const matrix::SubPanelView panel_heads_view(dist_heads_current, {0, 0}, nrefls_step);

    if (rank_panel == rank.col()) {
      const comm::IndexT_MPI rank_hoh = dist.template rank_global_tile<Coord::Row>(panel_offset.row());

      for (int idx_head = 0; idx_head < n_qr_heads; ++idx_head) {
        using dlaf::comm::schedule_bcast_recv;
        using dlaf::comm::schedule_bcast_send;

        const LocalTileIndex idx_panel_head(idx_head, 0);

        const GlobalTileIndex ij_head(panel_view.offset().row() + idx_head, j);
        const comm::IndexT_MPI rank_head = dist.template rank_global_tile<Coord::Row>(ij_head.row());

        if (rank.row() == rank_head) {
          // copy - set - send
          ex::start_detached(ex::when_all(mat_a.read(ij_head), panel_heads.readwrite(idx_panel_head)) |
                             di::transform(di::Policy<B>(), [=](const auto& head_in, auto&& head) {
                               // TODO FIXME change copy and if possible just lower
                               // matrix::internal::copy(head_in, head);
                               lapack::lacpy(blas::Uplo::General, head.size().rows(), head.size().cols(),
                                             head_in.ptr(), head_in.ld(), head.ptr(), head.ld());
                               lapack::laset(blas::Uplo::Lower, head.size().rows() - 1,
                                             head.size().cols(), T(0), T(0), head.ptr({1, 0}),
                                             head.ld());
                             }));
          ex::start_detached(schedule_bcast_send(mpi_col_chain.exclusive(),
                                                 panel_heads.read(idx_panel_head)));
        }
        else {
          // receive
          ex::start_detached(schedule_bcast_recv(mpi_col_chain.exclusive(), rank_head,
                                                 panel_heads.readwrite(idx_panel_head)));
        }
      }

      // QR local on heads
      red2band::local::computePanelReflectors(panel_heads, mat_taus_2nd, j, panel_heads_view);

      // copy back data
      {
        // - just head of heads upper to mat_a
        // - reflectors to hh_2nd
        const GlobalTileIndex ij_hoh(panel_view.offset().row(), j);
        if (rank.row() == dist.template rank_global_tile<Coord::Row>(ij_hoh.row()))
          ex::start_detached(ex::when_all(panel_heads.read({0, 0}), mat_a.readwrite(ij_hoh)) |
                             di::transform(di::Policy<B>(), [](const auto& hoh, auto&& hoh_a) {
                               common::internal::SingleThreadedBlasScope single;
                               lapack::lacpy(blas::Uplo::Upper, hoh.size().rows(), hoh.size().cols(),
                                             hoh.ptr(), hoh.ld(), hoh_a.ptr(), hoh_a.ld());
                             }));

        // Note: not all ranks might have an head
        if (!panel_view.iteratorLocal().empty()) {
          const auto i_head_lc =
              dist.template next_local_tile_from_global_tile<Coord::Row>(panel_view.offset().row());
          const auto i_head = dist.template global_tile_from_local_tile<Coord::Row>(i_head_lc);
          const auto idx_head = i_head - panel_view.offset().row();
          const LocalTileIndex idx_panel_head(idx_head, 0);
          const GlobalTileIndex ij_head(0, j);

          auto sender_heads =
              ex::when_all(panel_heads.read(idx_panel_head), mat_hh_2nd.readwrite(ij_head));

          if (rank.row() == rank_hoh) {
            ex::start_detached(std::move(sender_heads) |
                               di::transform(di::Policy<B>(), [](const auto& head, auto&& head_a) {
                                 common::internal::SingleThreadedBlasScope single;
                                 lapack::laset(blas::Uplo::Upper, head_a.size().rows(),
                                               head_a.size().cols(), T(0), T(1), head_a.ptr(),
                                               head_a.ld());
                                 lapack::lacpy(blas::Uplo::Lower, head.size().rows() - 1,
                                               head.size().cols() - 1, head.ptr({1, 0}), head.ld(),
                                               head_a.ptr({1, 0}), head_a.ld());
                               }));
          }
          else {
            ex::start_detached(std::move(sender_heads) |
                               di::transform(di::Policy<B>(), [](const auto& head, auto&& head_a) {
                                 common::internal::SingleThreadedBlasScope single;
                                 lapack::lacpy(blas::Uplo::General, head.size().rows(),
                                               head.size().cols(), head.ptr(), head.ld(), head_a.ptr(),
                                               head_a.ld());
                               }));
          }
        }
      }
    }

    // TRAILING 2nd pass
    {
      const GlobalTileIndex at_end_L(at_offset.row() + n_qr_heads, 0);
      const GlobalTileIndex at_end_R(0, at_offset.col() + n_qr_heads);

      const LocalTileIndex zero_lc(0, 0);
      matrix::Matrix<T, D> ws_T({nrefls_step, nrefls_step}, dist.block_size());

      auto& ws_V = panels_v.nextResource();
      ws_V.setRange(at_offset, at_end_L);
      ws_V.setWidth(nrefls_step);

      if (rank_panel == rank.col()) {
        // setup reflector panel
        const GlobalTileIndex ij_head(0, j);
        const LocalTileIndex ij_vhh_lc(ws_V.rangeStartLocal(), 0);

        if (!ws_V.iteratorLocal().empty()) {
          ex::start_detached(ex::when_all(mat_hh_2nd.read(ij_head), ws_V.readwrite(ij_vhh_lc)) |
                             di::transform(di::Policy<B>(), [=](const auto& head_in, auto&& head) {
                               lapack::lacpy(blas::Uplo::General, head.size().rows(), head.size().cols(),
                                             head_in.ptr(), head_in.ld(), head.ptr(), head.ld());
                             }));
        }

        for (const auto& i_lc : ws_V.iteratorLocal()) {
          std::ostringstream ss;
          ss << "V2nd(" << dist.global_tile_index(i_lc) << ")";
          print_sync(ss.str(), ws_V.read(i_lc));
        }

        using factorization::internal::computeTFactor;
        const GlobalTileIndex j_tau(j, 0);
        computeTFactor<B>(ws_V, mat_taus_2nd.read(j_tau), ws_T.readwrite(zero_lc), mpi_col_chain);

        print_sync("T2nd", ws_T.read(zero_lc));
      }

      auto& ws_VT = panels_vt.nextResource();
      ws_VT.setRange(at_offset, at_end_R);
      ws_VT.setHeight(nrefls_step);

      comm::broadcast(rank_panel, ws_V, ws_VT, mpi_row_chain, mpi_col_chain);

      for (const auto& i_lc : ws_VT.iteratorLocal()) {
        std::ostringstream ss;
        ss << "VT2nd(" << dist.global_tile_index(i_lc) << ")";
        print_sync(ss.str(), ws_VT.read(i_lc));
      }

      // Note:
      // Differently from 1st pass, where transformations are independent one from the other,
      // this 2nd pass is a single QR transformation that has to be applied from L and R.

      // W0 = V T
      auto& ws_W0 = panels_w0.nextResource();
      ws_W0.setRange(at_offset, at_end_L);
      ws_W0.setWidth(nrefls_step);

      if (rank.col() == rank_panel)
        red2band::local::trmmComputeW<B, D>(ws_W0, ws_V, ws_T.read(zero_lc));

      // distribute W0 -> W0T
      auto& ws_W0T = panels_w0t.nextResource();
      ws_W0T.setRange(at_offset, at_end_R);
      ws_W0T.setHeight(nrefls_step);

      comm::broadcast(rank_panel, ws_W0, ws_W0T, mpi_row_chain, mpi_col_chain);

      for (const auto& idx : ws_W0.iteratorLocal()) {
        std::ostringstream ss;
        ss << "W0(" << dist.global_tile_index(idx) << ")";
        print_sync(ss.str(), ws_W0.read(idx));
      }

      for (const auto& idx : ws_W0T.iteratorLocal()) {
        std::ostringstream ss;
        ss << "W0T(" << dist.global_tile_index(idx) << ")";
        print_sync(ss.str(), ws_W0T.read(idx));
      }

      // W1 = A W0
      auto& ws_W1 = panels_w1.nextResource();
      ws_W1.setRange(at_offset, at_end_L);
      ws_W1.setWidth(nrefls_step);

      auto& ws_W1T = panels_w1t.nextResource();
      ws_W1T.setRange(at_offset, at_end_R);
      ws_W1T.setHeight(nrefls_step);

      ca_red2band::hemm2nd<B, D>(rank_panel, ws_W1, ws_W1T, at_view, at_end_R.col(), mat_a, ws_W0,
                                 ws_W0T, mpi_row_chain, mpi_col_chain);

      for (const auto& idx : ws_W1.iteratorLocal()) {
        std::ostringstream ss;
        ss << "W1(" << dist.global_tile_index(idx) << ")";
        print_sync(ss.str(), ws_W1.read(idx));
      }

      // W1 = W1 - 0.5 V W0* W1
      if (rank.col() == rank_panel) {
        matrix::Matrix<T, D> ws_W2 = std::move(ws_T);

        // W2 = W0T W1
        red2band::local::gemmComputeW2<B, D>(ws_W2, ws_W0, ws_W1);
        if (mpi_col_chain.size() > 1) {
          ex::start_detached(comm::schedule_all_reduce_in_place(mpi_col_chain.exclusive(), MPI_SUM,
                                                                ws_W2.readwrite(zero_lc)));
        }

        print_sync("W2", ws_W2.read(zero_lc));

        // W1 = W1 - 0.5 V W2
        red2band::local::gemmUpdateX<B, D>(ws_W1, ws_W2, ws_V);
      }

      // distribute W1 -> W1T
      ws_W1T.reset();
      ws_W1T.setRange(at_offset, at_end_R);
      ws_W1T.setHeight(nrefls_step);

      comm::broadcast(rank_panel, ws_W1, ws_W1T, mpi_row_chain, mpi_col_chain);

      // A -= W1 VT + V W1T
      ca_red2band::her2k_2nd<B>(at_end_R.col(), at_view, mat_a, ws_W1, ws_W1T, ws_V, ws_VT);

      ws_W1T.reset();
      ws_W1.reset();
      ws_W0T.reset();
      ws_W0.reset();
      ws_VT.reset();
      ws_V.reset();
    }

    panel_heads.reset();
  }

  return {std::move(mat_taus_1st), std::move(mat_taus_2nd), std::move(mat_hh_2nd)};
}
}