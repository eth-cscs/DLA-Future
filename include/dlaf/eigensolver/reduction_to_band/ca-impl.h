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

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/kernels/broadcast.h>
#include <dlaf/eigensolver/reduction_to_band/api.h>
#include <dlaf/eigensolver/reduction_to_band/common.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/copy_tile.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/panel.h>
#include <dlaf/schedulers.h>
#include <dlaf/sender/transform.h>
#include <dlaf/sender/transform_mpi.h>

//
#include <iostream>

#include <dlaf/matrix/print_numpy.h>

namespace dlaf::eigensolver::internal {

namespace ca_red2band {}

// Distributed implementation of reduction to band
template <Backend B, Device D, class T>
CARed2BandResult<T, D> CAReductionToBand<B, D, T>::call(comm::CommunicatorGrid& grid,
                                                        Matrix<T, D>& mat_a, const SizeType band_size) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

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

  constexpr std::size_t n_workspaces = 2;

  // TODO HEADS workspace
  // - column vector
  // - has to be fully local
  // - no more than grid_size.rows() tiles (1 tile per rank in the column)
  // - we use panel just because it offers the ability to shrink width/height
  const matrix::Distribution dist_heads(
      LocalElementSize(dist.grid_size().rows() * dist.block_size().rows(), dist.block_size().cols()),
      dist.block_size());
  common::RoundRobin<matrix::Panel<Coord::Col, T, D>> panels_heads(n_workspaces, dist_heads);

  DLAF_ASSERT(mat_a.block_size().cols() == band_size, mat_a.block_size().cols(), band_size);
  const SizeType ntiles = (nrefls - 1) / band_size + 1;

  for (SizeType j = 0; j < ntiles; ++j) {
    const SizeType i = j + 1;
    // const SizeType j_lc = dist.template next_local_tile_from_global_tile<Coord::Col>(j);

    // panel
    const GlobalTileIndex panel_offset(i, j);
    const GlobalElementIndex panel_offset_el(panel_offset.row() * band_size,
                                             panel_offset.col() * band_size);
    matrix::SubPanelView panel_view(dist, panel_offset_el, band_size);
    const comm::IndexT_MPI rank_panel = dist.template rank_global_tile<Coord::Col>(panel_offset.col());

    // trailing
    const GlobalTileIndex trailing_offset(i, j + 1);
    const GlobalElementIndex trailing_offset_el(trailing_offset.row() * band_size,
                                                trailing_offset.col() * band_size);
    matrix::SubMatrixView(dist, trailing_offset_el);

    // PANEL: just ranks in the current column
    // QR local (HH reflectors stored in-place)
    if (rank_panel == rank.col()) {
      // Note:  SubPanelView is (at most) band_size wide, but it may contain a smaller number of
      //        reflectors (i.e. at the end when last reflector size is 1)
      red2band::local::computePanelReflectors(mat_a, mat_taus_1st, panel_offset.col(), panel_view);
    }

    // TRAILING 1st pass
    {}

    // PANEL: just ranks in the current column
    // QR local with just heads (HH reflectors have to be computed elsewhere, 1st-pass in-place)
    if (rank_panel == rank.col()) {
      auto&& panel_heads = panels_heads.nextResource();

      const SizeType i_begin_head = i;
      const SizeType i_end_head =
          std::min((i_begin_head + dist.grid_size().rows() + 1), dist.nr_tiles().rows());

      const SizeType nheads = i_end_head - i_begin_head;
      panel_heads.setRangeEnd({nheads, 0});

      for (SizeType i_head = i_begin_head, idx = 0; i_head < i_end_head; ++i_head, ++idx) {
        const LocalTileIndex idx_panel_head(idx, 0);
        const GlobalTileIndex ij_head(i_head, j);

        const comm::IndexT_MPI rank_owner = dist.template rank_global_tile<Coord::Row>(i_head);

        if (rank.row() == rank_owner) {
          // copy - set - send
          ex::start_detached(ex::when_all(mat_a.read(ij_head), panel_heads.readwrite(idx_panel_head)) |
                             di::transform(di::Policy<B>(), [=](const auto& head_in, auto&& head) {
                               matrix::internal::copy(head_in, head);
                               lapack::laset(blas::Uplo::Lower, head.size().rows() - 1,
                                             head.size().cols(), T(0), T(0), head.ptr({1, 0}),
                                             head.ld());
                             }));

          ex::start_detached(dlaf::comm::schedule_bcast_send(mpi_col_chain.exclusive(),
                                                             panel_heads.read(idx_panel_head)));
        }
        else {
          // receive
          ex::start_detached(dlaf::comm::schedule_bcast_recv(mpi_col_chain.exclusive(), rank_owner,
                                                             panel_heads.readwrite(idx_panel_head)));
        }
      }

      // for (auto ij : panel_heads.iteratorLocal()) {
      //   const auto& tile = pika::this_thread::experimental::sync_wait(panel_heads.read(ij)).get();
      //   std::cout << "heads(" << ij << ") = ";
      //   print(format::numpy{}, tile);
      // }

      // QR local on heads
      // TODO check if some head tiles are not used and limit (or set to zero as a starting point)
      matrix::SubPanelView panel_view(dist_heads, {0, 0}, band_size);
      red2band::local::computePanelReflectors(panel_heads, mat_taus_2nd, 0, panel_view);

      for (auto ij : panel_heads.iteratorLocal()) {
        const auto& tile = pika::this_thread::experimental::sync_wait(panel_heads.read(ij)).get();
        std::cout << "heads_u(" << ij << ") = ";
        print(format::numpy{}, tile);
      }

      // TODO copy back data
      // - just head of heads upper to mat_a
      // - reflectors to hh_2nd

      const GlobalTileIndex ij_hoh(i_begin_head, j);
      if (rank.row() == dist.template rank_global_tile<Coord::Row>(ij_hoh.row()))
        ex::start_detached(ex::when_all(panel_heads.read({0, 0}), mat_a.readwrite(ij_hoh)) |
                           di::transform(di::Policy<B>(), [](const auto& hoh, auto&& hoh_a) {
                             common::internal::SingleThreadedBlasScope single;
                             lapack::lacpy(blas::Uplo::Upper, hoh.size().rows(), hoh.size().cols(),
                                           hoh.ptr(), hoh.ld(), hoh_a.ptr(), hoh_a.ld());
                             // std::cout << "hoh_a = ";
                             // print(format::numpy{}, hoh_a);
                           }));

      for (SizeType i_head = i_begin_head, idx = 0; i_head < i_end_head; ++i_head, ++idx) {
        const comm::IndexT_MPI rank_owner = dist.template rank_global_tile<Coord::Row>(i_head);

        if (rank.row() == rank_owner) {
          const LocalTileIndex idx_panel_head(idx, 0);
          const GlobalTileIndex ij_hh_2nd(0, j);
          const bool is_hoh = (idx == 0);

          auto sender_heads =
              ex::when_all(panel_heads.read(idx_panel_head), mat_hh_2nd.readwrite(ij_hh_2nd));

          if (is_hoh)
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
          else
            ex::start_detached(std::move(sender_heads) |
                               di::transform(di::Policy<B>(), [](const auto& head, auto&& head_a) {
                                 common::internal::SingleThreadedBlasScope single;
                                 lapack::lacpy(blas::Uplo::General, head.size().rows(),
                                               head.size().cols(), head.ptr(), head.ld(), head_a.ptr(),
                                               head_a.ld());
                               }));
        }
      }

      panel_heads.reset();
    }

    // TRAILING 2nd pass
    {}

    break;
  }

  return {std::move(mat_taus_1st), std::move(mat_taus_2nd), std::move(mat_hh_2nd)};
}
}
