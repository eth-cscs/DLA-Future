//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <utility>

#pragma once

#include <pika/execution.hpp>
#include <pika/thread.hpp>

#include <dlaf/blas/tile.h>
#include <dlaf/common/index2d.h>
#include <dlaf/common/pipeline.h>
#include <dlaf/common/round_robin.h>
#include <dlaf/communication/broadcast_panel.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/kernels.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/panel.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/multiplication/hermitian/api.h>
#include <dlaf/sender/when_all_lift.h>
#include <dlaf/util_matrix.h>

namespace dlaf::multiplication::internal {

namespace hermitian_ll {
template <Backend B, class T, typename ASender, typename BSender, typename CSender>
void hemm(const T alpha, ASender&& a_tile, BSender&& b_tile, const T beta, CSender&& c_tile) {
  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Side::Left, blas::Uplo::Lower, alpha,
                                  std::forward<ASender>(a_tile), std::forward<BSender>(b_tile), beta,
                                  std::forward<CSender>(c_tile)) |
      tile::hemm(dlaf::internal::Policy<B>(pika::execution::thread_priority::normal)));
}

template <Backend B, class T, typename ASender, typename BSender, typename CSender>
void gemmN(const T alpha, ASender&& a_tile, BSender&& b_tile, const T beta, CSender&& c_tile) {
  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, alpha,
                                  std::forward<ASender>(a_tile), std::forward<BSender>(b_tile), beta,
                                  std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<B>(pika::execution::thread_priority::normal)));
}

template <Backend B, class T, typename ASender, typename BSender, typename CSender>
void gemmC(const T alpha, ASender&& a_tile, BSender&& b_tile, const T beta, CSender&& c_tile) {
  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Op::ConjTrans, blas::Op::NoTrans, alpha,
                                  std::forward<ASender>(a_tile), std::forward<BSender>(b_tile), beta,
                                  std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<B>(pika::execution::thread_priority::normal)));
}
}

template <Backend B, Device D, class T>
void Hermitian<B, D, T>::call_LL(const T alpha, Matrix<const T, D>& mat_a, Matrix<const T, D>& mat_b,
                                 const T beta, Matrix<T, D>& mat_c) {
  using namespace hermitian_ll;
  const SizeType k = mat_a.distribution().localNrTiles().cols();

  for (const auto ij : common::iterate_range2d(mat_c.distribution().localNrTiles())) {
    T beta_ij = beta;
    for (SizeType l = 0; l < ij.row(); ++l) {
      auto il = LocalTileIndex{ij.row(), l};
      auto lj = LocalTileIndex{l, ij.col()};
      gemmN<B>(alpha, mat_a.read(il), mat_b.read(lj), beta_ij, mat_c.readwrite(ij));
      beta_ij = 1;
    }

    {
      auto ii = LocalTileIndex{ij.row(), ij.row()};
      hemm<B>(alpha, mat_a.read(ii), mat_b.read(ij), beta_ij, mat_c.readwrite(ij));
      beta_ij = 1;
    }

    for (SizeType l = ij.row() + 1; l < k; ++l) {
      auto li = LocalTileIndex{l, ij.row()};
      auto lj = LocalTileIndex{l, ij.col()};
      gemmC<B>(alpha, mat_a.read(li), mat_b.read(lj), beta_ij, mat_c.readwrite(ij));
      beta_ij = 1;
    }
  }
}

template <Backend B, Device D, class T>
void Hermitian<B, D, T>::call_LL(comm::CommunicatorGrid& grid, const T alpha, Matrix<const T, D>& mat_a,
                                 Matrix<const T, D>& mat_b, const T beta, Matrix<T, D>& mat_c) {
  using namespace hermitian_ll;
  namespace ex = pika::execution::experimental;
  using pika::execution::thread_priority;

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr_a = mat_a.distribution();
  const matrix::Distribution& distr_b = mat_b.distribution();
  const matrix::Distribution& distr_c = mat_c.distribution();

  if (mat_b.size().isEmpty())
    return;

  auto mpi_row_task_chain = grid.row_communicator_pipeline();
  auto mpi_col_task_chain = grid.col_communicator_pipeline();

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, D>> a_panels(n_workspaces, distr_a);
  common::RoundRobin<matrix::Panel<Coord::Row, T, D>> b_panels(n_workspaces, distr_b);
  common::RoundRobin<matrix::Panel<Coord::Row, T, D>> c_panels(n_workspaces, distr_c);

  const SizeType m_loc = distr_c.localNrTiles().rows();
  const SizeType n_loc = distr_c.localNrTiles().cols();
  const SizeType k = distr_a.nrTiles().cols();
  T beta_ = beta;

  for (SizeType l = 0; l < k; ++l) {
    const GlobalTileIndex ll{l, l};
    const LocalTileIndex ll_offset{distr_a.nextLocalTileFromGlobalTile<Coord::Row>(ll.row()),
                                   distr_a.nextLocalTileFromGlobalTile<Coord::Col>(ll.col())};

    const SizeType l_loc = distr_c.nextLocalTileFromGlobalTile<Coord::Row>(ll.row());
    const SizeType l1_loc = distr_c.nextLocalTileFromGlobalTile<Coord::Row>(ll.row() + 1);
    const LocalTileIndex diag_offset(l_loc, 0);
    const LocalTileIndex diag_end_offset(l1_loc, n_loc);
    const LocalTileIndex lower_offset(l1_loc, 0);
    const LocalTileIndex lower_end_offset(m_loc, n_loc);

    auto& a_panel = a_panels.nextResource();
    auto& b_panel = b_panels.nextResource();
    auto& c_panel = c_panels.nextResource();

    a_panel.setRangeStart(ll);

    if (l == mat_a.nrTiles().rows() - 1) {
      a_panel.setWidth(mat_a.tileSize(ll).cols());
      b_panel.setHeight(mat_a.tileSize(ll).cols());
      c_panel.setHeight(mat_a.tileSize(ll).cols());
    }

    const auto rank_ll = distr_a.rankGlobalTile(ll);
    if (this_rank.col() == rank_ll.col()) {
      for (SizeType i_loc = ll_offset.row(); i_loc < distr_a.localNrTiles().rows(); ++i_loc) {
        const LocalTileIndex il{i_loc, ll_offset.col()};
        a_panel.setTile(il, mat_a.read(il));
      }
    }
    comm::broadcast(rank_ll.col(), a_panel, mpi_row_task_chain);

    if (this_rank.row() == rank_ll.row()) {
      for (SizeType j_loc = 0; j_loc < distr_b.localNrTiles().cols(); ++j_loc) {
        const LocalTileIndex lj{diag_offset.row(), j_loc};
        b_panel.setTile(lj, mat_b.read(lj));
      }
    }
    comm::broadcast(rank_ll.row(), b_panel, mpi_col_task_chain);

    for (const auto ij : common::iterate_range2d(diag_offset, diag_end_offset)) {
      hemm<B>(alpha, a_panel.read(ij), b_panel.read(ij), beta_, mat_c.readwrite(ij));
    }

    if (l < k - 1) {
      matrix::util::set0<B>(thread_priority::normal, c_panel);
      // Note: As A is square, B and C have the same size and blocksize
      for (const auto ij : common::iterate_range2d(lower_offset, lower_end_offset)) {
        // No Transpose part
        // C_ij += A_ik * B_kj
        gemmN<B>(alpha, a_panel.read(ij), b_panel.read(ij), beta_, mat_c.readwrite(ij));

        // ConjTranspose part
        // C_kj += (A_ik)^T * B_ij
        gemmC<B>(alpha, a_panel.read(ij), mat_b.read(ij), T{1}, c_panel.readwrite(ij));
      }

      if (grid.colCommunicator().size() != 1) {
        for (const auto& idx : c_panel.iteratorLocal()) {
          if (this_rank.row() == rank_ll.row()) {
            ex::start_detached(comm::schedule_reduce_recv_in_place(mpi_col_task_chain.exclusive(),
                                                                   MPI_SUM, c_panel.readwrite(idx)));
          }
          else {
            ex::start_detached(comm::schedule_reduce_send(mpi_col_task_chain.exclusive(), rank_ll.row(),
                                                          MPI_SUM, c_panel.read(idx)));
          }
        }
      }
      for (const auto lj : common::iterate_range2d(diag_offset, diag_end_offset)) {
        pika::execution::experimental::start_detached(
            dlaf::internal::whenAllLift(T{1}, c_panel.read(lj), mat_c.readwrite(lj)) |
            tile::add(dlaf::internal::Policy<B>(thread_priority::high)));
      }
    }

    // First iteration scales all the C tiles.
    beta_ = T{1};

    a_panel.reset();
    b_panel.reset();
    c_panel.reset();
  }
}

}
