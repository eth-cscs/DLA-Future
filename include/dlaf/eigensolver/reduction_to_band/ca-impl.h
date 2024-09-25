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

#include <cstddef>
#include <utility>

#include <pika/execution.hpp>

#include <dlaf/common/index2d.h>
#include <dlaf/communication/broadcast_panel.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/index.h>
#include <dlaf/communication/kernels/all_reduce.h>
#include <dlaf/communication/kernels/broadcast.h>
#include <dlaf/eigensolver/reduction_to_band/api.h>
#include <dlaf/eigensolver/reduction_to_band/common.h>
#include <dlaf/factorization/qr.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/copy_tile.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/distribution_extensions.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/panel.h>
#include <dlaf/schedulers.h>
#include <dlaf/sender/transform.h>
#include <dlaf/sender/transform_mpi.h>

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
        // TODO fix doc
        // Note:
        // - We are updating from both L and R.
        // - We are computing all combinations of W3 and V (and viceversa), and putting results in A
        // - By looping on position of A that will contain the result
        // - We use the same row for the first operand
        // - We use the col as the row for the second operand
        const SizeType iT_lc = dist.template local_tile_from_global_tile<Coord::Row>(ij.col());

        // A -= W3 . V*
        her2kOffDiag<B>(priority, W3.read(ij_lc), V.read({iT_lc, 0}), getSubA());
        // A -= V . W3*
        her2kOffDiag<B>(priority, V.read(ij_lc), W3.read({iT_lc, 0}), getSubA());
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

      // skip upper
      if (ij.row() < ij.col()) {
        continue;
      }

      const bool is_diag = (ij.row() == ij.col());

      if (is_diag) {
        hemmDiag<B>(thread_priority::high, A.read(ij_lc), W0.read(ij_lc), W1.readwrite(ij_lc));
      }
      else {
        // Lower
        hemmOffDiag<B>(thread_priority::high, blas::Op::NoTrans, A.read(ij_lc), W0T.read(ij_lc),
                       W1.readwrite(ij_lc));

        // Upper
        const GlobalTileIndex ijU = transposed(ij);

        // Note: if it is out of the "sub-matrix"
        if (ijU.col() >= j_end)
          continue;

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
void her2k_2nd(const SizeType i_end, const SizeType j_end, const matrix::SubMatrixView& at_view,
               matrix::Matrix<T, D>& a, matrix::Panel<Coord::Col, const T, D>& W1,
               matrix::Panel<Coord::Row, T, D, matrix::StoreTransposed::Yes>& W1T,
               matrix::Panel<Coord::Col, const T, D>& V,
               matrix::Panel<Coord::Row, T, D, matrix::StoreTransposed::Yes>& VT) {
  static_assert(std::is_signed_v<BaseType<T>>, "alpha in computations requires to be -1");

  using pika::execution::thread_priority;
  using red2band::her2kDiag;
  using red2band::her2kOffDiag;

  const auto dist = a.distribution();

  const LocalTileIndex at_offset_lc = at_view.begin();

  const SizeType iL_end_lc = dist.template next_local_tile_from_global_tile<Coord::Row>(i_end);
  const SizeType jR_end_lc = dist.template next_local_tile_from_global_tile<Coord::Col>(j_end);
  for (SizeType i_lc = at_offset_lc.row(); i_lc < iL_end_lc; ++i_lc) {
    const auto j_end_lc =
        std::min(jR_end_lc, dist.template next_local_tile_from_global_tile<Coord::Col>(
                                dist.template global_tile_from_local_tile<Coord::Row>(i_lc) + 1));
    for (SizeType j_lc = at_offset_lc.col(); j_lc < j_end_lc; ++j_lc) {
      const LocalTileIndex ij_local{i_lc, j_lc};
      const GlobalTileIndex ij = dist.globalTileIndex(ij_local);

      const bool is_diagonal_tile = (ij.row() == ij.col());

      auto getSubA = [&a, &at_view, ij_local]() {
        return splitTile(a.readwrite(ij_local), at_view(ij_local));
      };

      // The first column of the trailing matrix (except for the very first global tile) has to be
      // updated first, in order to unlock the next iteration as soon as possible.
      const auto priority =
          (j_lc == at_offset_lc.col()) ? thread_priority::high : thread_priority::normal;

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

  // This is just going to update rows that are going to be updated just from right.
  for (SizeType i_lc = iL_end_lc; i_lc < dist.local_nr_tiles().rows(); ++i_lc) {
    const auto j_end_lc =
        std::min(jR_end_lc, dist.template next_local_tile_from_global_tile<Coord::Col>(
                                dist.template global_tile_from_local_tile<Coord::Row>(i_lc) + 1));
    for (SizeType j_lc = at_offset_lc.col(); j_lc < j_end_lc; ++j_lc) {
      const LocalTileIndex ij_lc{i_lc, j_lc};

      auto getSubA = [&a, &at_view, ij_lc]() { return splitTile(a.readwrite(ij_lc), at_view(ij_lc)); };

      // The first column of the trailing matrix (except for the very first global tile) has to be
      // updated first, in order to unlock the next iteration as soon as possible.
      const auto priority =
          (j_lc == at_offset_lc.col()) ? thread_priority::high : thread_priority::normal;

      // A -= X . V*
      her2kOffDiag<B>(priority, W1.read(ij_lc), VT.read(ij_lc), getSubA());
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

  // Note:
  // Reflector of size = 1 is not considered whatever T is (i.e. neither real nor complex)
  const SizeType nrefls = std::max<SizeType>(0, dist.size().cols() - band_size - 1);

  // Note:
  // Each rank has space for storing taus for one tile. It is distributed as the input matrix (i.e. 2D)
  // Note:
  // It is distributed "transposed" because of implicit assumptions in functions, i.e.
  // computePanelReflectors and computeTFactor, which expect a column vector.
  DLAF_ASSERT(dist.block_size().cols() % band_size == 0, dist.block_size().cols(), band_size);
  Matrix<T, Device::CPU> mat_taus_1st(matrix::Distribution(
      GlobalElementSize(nrefls, dist.grid_size().rows()), TileElementSize(dist.block_size().cols(), 1),
      transposed(dist.grid_size()), transposed(rank), transposed(dist.source_rank_index())));

  // Note:
  // It has room for storing one tile per rank and it is distributed as the input matrix (i.e. 2D)
  const matrix::Distribution dist_hh_2nd(
      GlobalElementSize(dist.grid_size().rows() * dist.block_size().rows(), dist.size().cols()),
      dist.block_size(), dist.grid_size(), rank, dist.source_rank_index());
  Matrix<T, D> mat_hh_2nd(dist_hh_2nd);

  // Note:
  // Is stored as a column but it acts as a row vector. It is replicated over rows and it is distributed
  // over columns (i.e. 1D distributed)
  DLAF_ASSERT(dist.block_size().cols() % band_size == 0, dist.block_size().cols(), band_size);
  Matrix<T, Device::CPU> mat_taus_2nd(
      matrix::Distribution(GlobalElementSize(nrefls, 1), TileElementSize(dist.block_size().cols(), 1),
                           comm::Size2D(dist.grid_size().cols(), 1), comm::Index2D(rank.col(), 0),
                           comm::Index2D(dist.source_rank_index().col(), 0)));

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
  DLAF_ASSERT(is_full_band, is_full_band);

  for (SizeType j = 0; j < ntiles; ++j) {
    const SizeType i = j + 1;
    const SizeType j_lc = dist.template local_tile_from_global_tile<Coord::Col>(j);

    const SizeType nrefls_1st = [&]() -> SizeType {
      const SizeType i_head_lc = dist.template next_local_tile_from_global_tile<Coord::Row>(i);

      if (i_head_lc >= dist.local_nr_tiles().rows())
        return 0;

      const SizeType i_head = dist.template global_tile_from_local_tile<Coord::Row>(i_head_lc);
      const GlobalTileIndex ij_head(i_head, j);
      const TileElementSize head_size = mat_a.tile_size_of(ij_head);

      if (i_head_lc == dist.local_nr_tiles().rows() - 1)
        return head_size.rows() - 1;
      else
        return head_size.rows();
    }();

    auto get_tile_tau = [&]() {
      if (nrefls_1st == band_size)
        return mat_taus_1st.readwrite(LocalTileIndex(j_lc, 0));
      return splitTile(mat_taus_1st.readwrite(LocalTileIndex(j_lc, 0)), {{0, 0}, {nrefls_1st, 1}});
    };

    auto get_tile_tau_ro = [&]() {
      if (nrefls_1st == band_size)
        return mat_taus_1st.read(LocalTileIndex(j_lc, 0));
      return splitTile(mat_taus_1st.read(LocalTileIndex(j_lc, 0)), {{0, 0}, {nrefls_1st, 1}});
    };

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
    if (rank_panel == rank.col()) {
      using red2band::local::computePanelReflectors;
      computePanelReflectors(mat_a, get_tile_tau(), panel_view);
    }

    // TRAILING 1st pass
    if (at_offset_el.isIn(mat_a.size())) {
      // TODO FIXME workaround for possibly non-participating ranks (they cannot have width = 0)
      const SizeType nrefls_1st_min = std::max<SizeType>(nrefls_1st, 1);

      const LocalTileIndex zero_lc(0, 0);
      matrix::Matrix<T, D> ws_T({nrefls_1st_min, nrefls_1st_min}, dist.block_size());

      auto& ws_V = panels_v.nextResource();
      ws_V.setRangeStart(at_offset);
      ws_V.setWidth(nrefls_1st_min);

      auto& ws_W0 = panels_w0.nextResource();
      ws_W0.setRangeStart(at_offset);
      ws_W0.setWidth(nrefls_1st_min);

      if (rank_panel == rank.col() && nrefls_1st != 0) {
        using factorization::internal::computeTFactor;
        using red2band::local::setupReflectorPanelV;

        const bool has_head = !panel_view.iteratorLocal().empty();
        setupReflectorPanelV<B, D, T>(has_head, panel_view, nrefls_1st_min, ws_V, mat_a, !is_full_band);

        computeTFactor<B>(ws_V, get_tile_tau_ro(), ws_T.readwrite(zero_lc));

        // W = V T
        red2band::local::trmmComputeW<B, D>(ws_W0, ws_V, ws_T.read(zero_lc));
      }

      // Note: apply local transformations, one after the other
      // matrix::Matrix<T, D> ws_W2 = std::move(ws_T);

      for (int idx_qr_head = 0; idx_qr_head < n_qr_heads; ++idx_qr_head) {
        const SizeType head_qr = at_view.offset().row() + idx_qr_head;
        const comm::Index2D rank_qr(dist.template rank_global_tile<Coord::Row>(head_qr), rank_panel);

        const bool is_row_involved = rank_qr.row() == rank.row();
        const bool is_col_involved = [head_qr, dist, rank]() {
          for (SizeType k = head_qr; k < dist.nr_tiles().rows(); k += dist.grid_size().rows()) {
            const comm::IndexT_MPI rank_owner_col = dist.template rank_global_tile<Coord::Col>(k);
            if (rank_owner_col == rank.col())
              return true;
          }
          return false;
        }();

        const SizeType nrtiles_transf =
            util::ceilDiv<SizeType>(dist.nr_tiles().rows() - head_qr, dist.grid_size().rows());

        // TODO FIXME cannot skip because otherwise no W1 reduction if any rank skip?!
        // // this rank is not involved at all
        // if (!is_row_involved && !is_col_involved)
        //   continue;

        // number of reflectors of this "local" transformation
        const SizeType nrefls_this = [&]() {
          using matrix::internal::distribution::global_tile_from_local_tile_on_rank;
          const SizeType i_head = head_qr;
          if (nrtiles_transf == 1) {
            const TileElementSize tile_size = dist.tile_size_of({i_head, j});
            return std::min<SizeType>(tile_size.rows() - 1, tile_size.cols());
          }
          return dist.block_size().cols();
        }();

        if (nrefls_this == 0)
          continue;

        // "local" broadcast along rows involved in this local transformation
        if (is_row_involved) {
          comm::broadcast(rank_panel, ws_V, mpi_row_chain);
          comm::broadcast(rank_panel, ws_W0, mpi_row_chain);
        }

        auto& ws_VT = panels_vt.nextResource();
        // TODO FIXME workaround for panel problem on reset about range
        ws_VT.setRange(at_offset, common::indexFromOrigin(dist.nr_tiles()));
        ws_VT.setHeight(nrefls_this);

        auto& ws_W0T = panels_w0t.nextResource();
        // TODO FIXME workaround for panel problem on reset about range
        ws_W0T.setRange(at_offset, common::indexFromOrigin(dist.nr_tiles()));
        ws_W0T.setHeight(nrefls_this);

        // broadcast along cols involved in this local transformation
        if (is_col_involved) {  // diagonal
          // set diagonal tiles
          for (const auto ij_lc : ws_VT.iteratorLocal()) {
            const SizeType k = dist.template global_tile_from_local_tile<Coord::Col>(ij_lc.col());
            const comm::IndexT_MPI rank_src = dist.template rank_global_tile<Coord::Row>(k);

            if (rank_qr.row() != rank_src)
              continue;

            using comm::schedule_bcast_recv;
            using comm::schedule_bcast_send;

            if (rank_src == rank.row()) {
              const SizeType i_lc = dist.template local_tile_from_global_tile<Coord::Row>(k);

              ws_VT.setTile(ij_lc, ws_V.read({i_lc, 0}));
              ws_W0T.setTile(ij_lc, ws_W0.read({i_lc, 0}));

              // if (nrtiles_transf > 1) {
              ex::start_detached(schedule_bcast_send(mpi_col_chain.exclusive(), ws_V.read({i_lc, 0})));
              ex::start_detached(schedule_bcast_send(mpi_col_chain.exclusive(), ws_W0.read({i_lc, 0})));
              // }
            }
            else {
              // if (nrtiles_transf > 1) {
              ex::start_detached(schedule_bcast_recv(mpi_col_chain.exclusive(), rank_src,
                                                     ws_VT.readwrite(ij_lc)));
              ex::start_detached(schedule_bcast_recv(mpi_col_chain.exclusive(), rank_src,
                                                     ws_W0T.readwrite(ij_lc)));
              // }
            }
          }
        }

        // W1 = A W0
        // Note:
        // it will fill up all W1 (distributed) and it will read just part of A, i.e. columns where
        // this local transformation should be applied from the right.
        auto& ws_W1 = panels_w1.nextResource();
        auto& ws_W1T = panels_w1t.nextResource();
        ws_W1.setRangeStart(at_offset);
        ws_W1.setWidth(nrefls_this);

        ws_W1T.setRangeStart(at_offset);
        ws_W1T.setHeight(nrefls_this);

        // TODO FIXME restrict communication to just interested ones
        using ca_red2band::hemm;
        hemm<B>(rank_qr, ws_W1, ws_W1T, at_view, mat_a, ws_W0, ws_W0T, mpi_row_chain, mpi_col_chain);

        // Note:
        // W1T has been used as support panel, so reset it again.
        ws_W1T.reset();
        ws_W1T.setRangeStart(at_offset);
        ws_W1T.setHeight(nrefls_this);

        // TODO FIXME restrict communication to just interested ones
        comm::broadcast(rank_panel, ws_W1, ws_W1T, mpi_row_chain, mpi_col_chain);

        // LR
        // A -= V W1* + W1 V* - V W0* W1 V*
        if (rank.row() == rank_qr.row()) {
          matrix::Matrix<T, D> ws_W2({nrefls_this, nrefls_this}, dist.block_size());
          // W2 = W0.T W1
          red2band::local::gemmComputeW2<B, D>(ws_W2, ws_W0, ws_W1);

          // Note:
          // Next steps for L and R need W1, so we create a copy that we are going to update for this step.
          auto& ws_W3 = panels_w3.nextResource();
          ws_W3.setRangeStart(at_offset);
          ws_W3.setWidth(nrefls_this);

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
                                               ws_W1.read(ij_lc), ws_VT.read(ij_lc), T(1),
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

        ws_W0T.reset();
        ws_VT.reset();
      }

      ws_W0.reset();
      ws_V.reset();
    }

    // ===== 2nd pass
    const matrix::Distribution dist_heads_current = [&]() {
      using matrix::internal::distribution::global_tile_element_distance;
      const SizeType i_begin = i;
      const SizeType i_end = std::min<SizeType>(i + dist.grid_size().rows(), dist.nr_tiles().rows());
      const SizeType nrows = global_tile_element_distance<Coord::Row>(dist, i_begin, i_end);
      return matrix::Distribution{LocalElementSize(nrows, band_size), dist.block_size()};
    }();

    const SizeType nrefls_step = [&]() {
      const SizeType reflector_size = dist_heads_current.size().rows();
      return std::min(dist_heads_current.size().cols(), reflector_size - 1);
    }();

    auto get_tile_tau2 = [&]() {
      if (nrefls_step == band_size)
        return mat_taus_2nd.readwrite(LocalTileIndex(j_lc, 0));
      return splitTile(mat_taus_2nd.readwrite(LocalTileIndex(j_lc, 0)), {{0, 0}, {nrefls_step, 1}});
    };

    auto get_tile_tau2_ro = [&]() {
      if (nrefls_step == band_size)
        return mat_taus_2nd.read(LocalTileIndex(j_lc, 0));
      return splitTile(mat_taus_2nd.read(LocalTileIndex(j_lc, 0)), {{0, 0}, {nrefls_step, 1}});
    };

    // PANEL: just ranks in the current column
    // QR local with just heads (HH reflectors have to be computed elsewhere, 1st-pass in-place)
    auto&& panel_heads = panels_heads.nextResource();
    panel_heads.setRangeEnd({n_qr_heads, 0});

    const matrix::SubPanelView panel_heads_view(dist_heads_current, {0, 0}, band_size);

    const bool rank_has_head_row = !panel_view.iteratorLocal().empty();
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
      using red2band::local::computePanelReflectors;
      computePanelReflectors(panel_heads, get_tile_tau2(), panel_heads_view);

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
        if (rank_has_head_row) {
          const auto i_head_lc =
              dist.template next_local_tile_from_global_tile<Coord::Row>(panel_view.offset().row());
          const auto i_head = dist.template global_tile_from_local_tile<Coord::Row>(i_head_lc);
          const auto idx_head = i_head - panel_view.offset().row();
          const LocalTileIndex idx_panel_head(idx_head, 0);
          const LocalTileIndex ij_head(0, j_lc);

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

      panel_heads.reset();
    }

    // TRAILING 2nd pass
    {
      panel_heads.setRangeEnd({n_qr_heads, 0});
      panel_heads.setWidth(nrefls_step);

      const GlobalTileIndex at_end_L(at_offset.row() + n_qr_heads, 0);
      const GlobalTileIndex at_end_R(0, at_offset.col() + n_qr_heads);

      const LocalTileIndex zero_lc(0, 0);
      matrix::Matrix<T, D> ws_T({nrefls_step, nrefls_step}, dist.block_size());

      auto& ws_V = panels_v.nextResource();
      ws_V.setRange(at_offset, at_end_L);
      ws_V.setWidth(nrefls_step);

      if (rank_panel == rank.col()) {
        // setup reflector panel
        const LocalTileIndex ij_head(0, j_lc);
        const LocalTileIndex ij_vhh_lc(ws_V.rangeStartLocal(), 0);

        if (rank_has_head_row) {
          ex::start_detached(ex::when_all(mat_hh_2nd.read(ij_head), ws_V.readwrite(ij_vhh_lc)) |
                             di::transform(di::Policy<B>(), [=](const auto& head_in, auto&& head) {
                               lapack::lacpy(blas::Uplo::General, head.size().rows(), head.size().cols(),
                                             head_in.ptr(), head_in.ld(), head.ptr(), head.ld());
                             }));
        }

        using factorization::internal::computeTFactor;
        const GlobalTileIndex j_tau(j, 0);
        computeTFactor<B>(panel_heads, get_tile_tau2_ro(), ws_T.readwrite(zero_lc));
      }

      auto& ws_VT = panels_vt.nextResource();

      ws_VT.setRange(at_offset, at_end_R);
      ws_VT.setHeight(nrefls_step);

      comm::broadcast_all(rank_panel, ws_V, ws_VT, mpi_row_chain, mpi_col_chain);

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

      comm::broadcast_all(rank_panel, ws_W0, ws_W0T, mpi_row_chain, mpi_col_chain);

      // W1 = A W0
      auto& ws_W1 = panels_w1.nextResource();
      ws_W1.setRangeStart(at_offset);
      ws_W1.setWidth(nrefls_step);

      auto& ws_W1T = panels_w1t.nextResource();
      ws_W1T.setRangeStart(at_offset);
      ws_W1T.setHeight(nrefls_step);

      ca_red2band::hemm2nd<B, D>(rank_panel, ws_W1, ws_W1T, at_view, at_end_R.col(), mat_a, ws_W0,
                                 ws_W0T, mpi_row_chain, mpi_col_chain);

      // W1 = W1 - 0.5 V W0* W1
      if (rank.col() == rank_panel) {
        matrix::Matrix<T, D> ws_W2 = std::move(ws_T);

        // W2 = W0T W1
        red2band::local::gemmComputeW2<B, D>(ws_W2, ws_W0, ws_W1);
        if (mpi_col_chain.size() > 1) {
          ex::start_detached(comm::schedule_all_reduce_in_place(mpi_col_chain.exclusive(), MPI_SUM,
                                                                ws_W2.readwrite(zero_lc)));
        }

        // W1 = W1 - 0.5 V W2
        red2band::local::gemmUpdateX<B, D>(ws_W1, ws_W2, ws_V);
      }

      // distribute W1 -> W1T
      ws_W1T.reset();
      ws_W1T.setRangeStart(at_offset);
      ws_W1T.setHeight(nrefls_step);

      comm::broadcast(rank_panel, ws_W1, ws_W1T, mpi_row_chain, mpi_col_chain);

      // LR: A -= W1 VT + V W1T
      // R : [at_end_L.row():, :at_endR_col()] A = A - W1 V.T
      ca_red2band::her2k_2nd<B>(at_end_L.row(), at_end_R.col(), at_view, mat_a, ws_W1, ws_W1T, ws_V,
                                ws_VT);

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
