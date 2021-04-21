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

#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/util.hpp>

#include "dlaf/blas_tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/executors.h"
#include "dlaf/factorization/cholesky/api.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/panel.h"

#include "dlaf/memory/memory_view.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace factorization {
namespace internal {

template <class T>
void potrf_diag_tile(hpx::execution::parallel_executor executor_hp,
                     hpx::future<matrix::Tile<T, Device::CPU>> matrix_tile) {
  hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::potrf<T, Device::CPU>), blas::Uplo::Lower,
                std::move(matrix_tile));
}

template <class T>
void trsm_panel_tile(hpx::execution::parallel_executor executor_hp,
                     hpx::shared_future<matrix::Tile<const T, Device::CPU>> kk_tile,
                     hpx::future<matrix::Tile<T, Device::CPU>> matrix_tile) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), blas::Side::Right,
                blas::Uplo::Lower, blas::Op::ConjTrans, blas::Diag::NonUnit, T(1.0), std::move(kk_tile),
                std::move(matrix_tile));
}

template <class T>
void herk_trailing_diag_tile(hpx::execution::parallel_executor ex,
                             hpx::shared_future<matrix::Tile<const T, Device::CPU>> panel_tile,
                             hpx::future<matrix::Tile<T, Device::CPU>> matrix_tile) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::herk_o), blas::Uplo::Lower, blas::Op::NoTrans,
                BaseType<T>(-1.0), panel_tile, BaseType<T>(1.0), std::move(matrix_tile));
}

template <class T>
void gemm_trailing_matrix_tile(hpx::execution::parallel_executor ex,
                               hpx::shared_future<matrix::Tile<const T, Device::CPU>> panel_tile,
                               hpx::shared_future<matrix::Tile<const T, Device::CPU>> col_panel,
                               hpx::future<matrix::Tile<T, Device::CPU>> matrix_tile) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::NoTrans, blas::Op::ConjTrans,
                T(-1.0), std::move(panel_tile), std::move(col_panel), T(1.0), std::move(matrix_tile));
}

template <class T>
struct Cholesky<Backend::MC, Device::CPU, T> {
  static void call_L(Matrix<T, Device::CPU>& mat_a);
  static void call_L(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_a);
};

// Local implementation of Lower Cholesky factorization.
template <class T>
void Cholesky<Backend::MC, Device::CPU, T>::call_L(Matrix<T, Device::CPU>& mat_a) {
  auto executor_hp = dlaf::getHpExecutor<Backend::MC>();
  auto executor_np = dlaf::getNpExecutor<Backend::MC>();

  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().cols();

  for (SizeType k = 0; k < nrtile; ++k) {
    // Cholesky decomposition on mat_a(k,k) r/w potrf (lapack operation)
    auto kk = LocalTileIndex{k, k};

    potrf_diag_tile(executor_hp, mat_a(kk));

    for (SizeType i = k + 1; i < nrtile; ++i) {
      // Update panel mat_a(i,k) with trsm (blas operation), using data mat_a.read(k,k)
      trsm_panel_tile(executor_hp, mat_a.read(kk), mat_a(LocalTileIndex{i, k}));
    }

    for (SizeType j = k + 1; j < nrtile; ++j) {
      // first trailing panel gets high priority (look ahead).
      auto& trailing_matrix_executor = (j == k + 1) ? executor_hp : executor_np;

      // Update trailing matrix: diagonal element mat_a(j,j), reading mat_a.read(j,k), using herk (blas operation)
      herk_trailing_diag_tile(trailing_matrix_executor, mat_a.read(LocalTileIndex{j, k}),
                              mat_a(LocalTileIndex{j, j}));

      for (SizeType i = j + 1; i < nrtile; ++i) {
        // Update remaining trailing matrix mat_a(i,j), reading mat_a.read(i,k) and mat_a.read(j,k),
        // using gemm (blas operation)
        gemm_trailing_matrix_tile(trailing_matrix_executor, mat_a.read(LocalTileIndex{i, k}),
                                  mat_a.read(LocalTileIndex{j, k}), mat_a(LocalTileIndex{i, j}));
      }
    }
  }
}

template <class T>
struct RoundRobin {
  template <class... Args>
  RoundRobin(std::size_t n, Args&&... args) : next_index_(0) {
    for (std::size_t i = 0; i < n; ++i)
      pool_.emplace_back(std::forward<Args>(args)...);
  }

  T& next_resource() {
    auto idx = (next_index_ + 1) % pool_.size();
    std::swap(idx, next_index_);
    return pool_[idx];
  }

  std::size_t next_index_;
  std::vector<T> pool_;
};

template <class T>
void Cholesky<Backend::MC, Device::CPU, T>::call_L(comm::CommunicatorGrid grid,
                                                   Matrix<T, Device::CPU>& mat_a) {
  using hpx::util::unwrapping;
  using hpx::dataflow;

  auto executor_hp = dlaf::getHpExecutor<Backend::MC>();
  auto executor_np = dlaf::getNpExecutor<Backend::MC>();

  // Set up MPI executor pipelines
  comm::Executor executor_mpi;
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator());
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator());

  const comm::Index2D this_rank = grid.rank();
  const comm::Size2D grid_size = grid.size();

  const matrix::Distribution& distr = mat_a.distribution();
  const SizeType nrtile = mat_a.nrTiles().cols();

  constexpr std::size_t N_WORKSPACES = 2;
  RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panel_cols(N_WORKSPACES, distr);
  RoundRobin<matrix::Panel<Coord::Row, T, Device::CPU>> panel_cols_t(N_WORKSPACES, distr);

  for (SizeType k = 0; k < nrtile; ++k) {
    const GlobalTileIndex kk_idx(k, k);
    const comm::Index2D kk_rank = distr.rankGlobalTile(kk_idx);

    const LocalTileSize kk_offset{
        distr.nextLocalTileFromGlobalTile<Coord::Row>(k),
        distr.nextLocalTileFromGlobalTile<Coord::Col>(k),
    };

    const LocalTileIndex diag_wp_idx{0, kk_offset.cols()};

    auto& panel_col = panel_cols.next_resource();
    auto& panel_col_t = panel_cols_t.next_resource();

    panel_col.set_offset(kk_offset);
    panel_col_t.set_offset(kk_offset);

    // Factorization of diagonal tile and broadcast it along the `k-th column
    if (kk_rank.col() == this_rank.col()) {
      if (kk_rank.row() == this_rank.row()) {
        potrf_diag_tile(executor_hp, mat_a(kk_idx));
        if (k != nrtile - 1) {
          panel_col_t.set_tile(diag_wp_idx, mat_a.read(kk_idx));
          dataflow(executor_mpi, matrix::unwrapExtendTiles(comm::sendBcast<T>), panel_col_t.read(diag_wp_idx),
                   mpi_col_task_chain());
        }
      }
      else {
        if (k != nrtile - 1) {
          dataflow(executor_mpi, unwrapping(comm::recvBcast<T>), panel_col_t(diag_wp_idx), kk_rank.row(),
                   mpi_col_task_chain());
        }
      }
    }

    if (k == nrtile - 1)
      continue;

    // COLUMN UPDATE
    for (SizeType i = distr.nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
         i < distr.localNrTiles().rows(); ++i) {
      const LocalTileIndex local_idx(Coord::Row, i);
      const LocalTileIndex ik_idx(i, distr.localTileFromGlobalTile<Coord::Col>(k));

      if (kk_rank.col() == this_rank.col()) {
        trsm_panel_tile(executor_hp, panel_col_t.read(diag_wp_idx), mat_a(ik_idx));

        panel_col.set_tile(local_idx, mat_a.read(ik_idx));
      }
    }

    // row panel has been used for temporary storage of diagonal panel for column update
    panel_col_t.reset();

    // TODO skip last step tile
    broadcast(executor_mpi, kk_rank.col(), panel_col, panel_col_t, mpi_row_task_chain,
              mpi_col_task_chain, grid_size);

    // TRAILING MATRIX
    for (SizeType jt_idx = k + 1; jt_idx < nrtile; ++jt_idx) {
      const auto owner = distr.rankGlobalTile({jt_idx, jt_idx});

      if (owner.col() != this_rank.col())
        continue;

      const auto j = distr.localTileFromGlobalTile<Coord::Col>(jt_idx);
      auto& trailing_matrix_executor = (j == k + 1) ? executor_hp : executor_np;
      if (this_rank.row() == owner.row()) {
        const auto i = distr.localTileFromGlobalTile<Coord::Row>(jt_idx);

        herk_trailing_diag_tile(trailing_matrix_executor, panel_col.read({Coord::Row, i}),
                                mat_a(LocalTileIndex{i, j}));
      }

      for (SizeType i_idx = jt_idx + 1; i_idx < nrtile; ++i_idx) {
        const auto owner_row = distr.rankGlobalTile<Coord::Row>(i_idx);

        if (owner_row != this_rank.row())
          continue;

        const auto i = distr.localTileFromGlobalTile<Coord::Row>(i_idx);
        gemm_trailing_matrix_tile(executor_np, panel_col.read({Coord::Row, i}),
                                  panel_col_t.read({Coord::Col, j}), mat_a(LocalTileIndex{i, j}));
      }
    }

    panel_col.reset();
    panel_col_t.reset();
  }
}

/// ---- ETI
#define DLAF_CHOLESKY_MC_ETI(KWORD, DATATYPE) \
  KWORD template struct Cholesky<Backend::MC, Device::CPU, DATATYPE>;

DLAF_CHOLESKY_MC_ETI(extern, float)
DLAF_CHOLESKY_MC_ETI(extern, double)
DLAF_CHOLESKY_MC_ETI(extern, std::complex<float>)
DLAF_CHOLESKY_MC_ETI(extern, std::complex<double>)
}
}
}
