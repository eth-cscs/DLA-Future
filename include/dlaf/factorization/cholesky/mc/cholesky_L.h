//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/util/annotated_function.hpp>

#include <unordered_map>

#include "dlaf/blas_tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace internal {
namespace mc {

// If the diagonal tile is on this node factorize it
// Cholesky decomposition on mat_a(k,k) r/w potrf (lapack operation)
template <class T>
void potrf_diag_tile(hpx::threads::executors::pool_executor executor_hp,
                     hpx::future<matrix::Tile<T, Device::CPU>> matrix_tile) {
  hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::potrf<T, Device::CPU>), blas::Uplo::Lower,
                std::move(matrix_tile));
}

template <class T>
void trsm_panel_tile(hpx::threads::executors::pool_executor executor_hp,
                     hpx::shared_future<matrix::Tile<const T, Device::CPU>> kk_tile,
                     hpx::future<matrix::Tile<T, Device::CPU>> matrix_tile) {
  hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), blas::Side::Right,
                blas::Uplo::Lower, blas::Op::ConjTrans, blas::Diag::NonUnit, 1.0, std::move(kk_tile),
                std::move(matrix_tile));
}

template <class T>
void herk_trailing_diag_tile(hpx::threads::executors::pool_executor trailing_matrix_executor,
                             hpx::shared_future<matrix::Tile<const T, Device::CPU>> panel_tile,
                             hpx::future<matrix::Tile<T, Device::CPU>> matrix_tile) {
  hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::herk<T, Device::CPU>),
                blas::Uplo::Lower, blas::Op::NoTrans, -1.0, panel_tile, 1.0, std::move(matrix_tile));
}

template <class T>
void gemm_trailing_matrix_tile(hpx::threads::executors::pool_executor trailing_matrix_executor,
                               hpx::shared_future<matrix::Tile<const T, Device::CPU>> panel_tile,
                               hpx::shared_future<matrix::Tile<const T, Device::CPU>> col_panel,
                               hpx::future<matrix::Tile<T, Device::CPU>> matrix_tile) {
  hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>),
                blas::Op::NoTrans, blas::Op::ConjTrans, -1.0, std::move(panel_tile),
                std::move(col_panel), 1.0, std::move(matrix_tile));
}

template <class T>
void send_tile(hpx::threads::executors::pool_executor ex,
               common::Pipeline<comm::CommunicatorGrid>& task_chain, Coord rc_comm,
               hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile) {
  using ConstTile_t = matrix::Tile<const T, Device::CPU>;
  using PromiseComm_t = common::PromiseGuard<comm::CommunicatorGrid>;

  // Broadcast the (trailing) panel column-wise
  auto send_bcast_f = hpx::util::annotated_function(
      [rc_comm](hpx::shared_future<ConstTile_t> ftile, hpx::future<PromiseComm_t> fpcomm) {
        PromiseComm_t pcomm = fpcomm.get();
        comm::sync::broadcast::send(pcomm.ref().subCommunicator(rc_comm), ftile.get());
      },
      "send_tile");
  hpx::dataflow(ex, std::move(send_bcast_f), tile, task_chain());
}

template <class T>
hpx::future<matrix::Tile<const T, Device::CPU>> recv_tile(
    hpx::threads::executors::pool_executor ex, common::Pipeline<comm::CommunicatorGrid>& mpi_task_chain,
    Coord rc_comm, TileElementSize tile_size, int rank) {
  using ConstTile_t = matrix::Tile<const T, Device::CPU>;
  using PromiseComm_t = common::PromiseGuard<comm::CommunicatorGrid>;
  using MemView_t = memory::MemoryView<T, Device::CPU>;
  using Tile_t = matrix::Tile<T, Device::CPU>;

  // Update the (trailing) panel column-wise
  auto recv_bcast_f = hpx::util::annotated_function(
      [rank, tile_size, rc_comm](hpx::future<PromiseComm_t> fpcomm) -> ConstTile_t {
        PromiseComm_t pcomm = fpcomm.get();
        MemView_t mem_view(tile_size.linear_size());
        Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());
        comm::sync::broadcast::receive_from(rank, pcomm.ref().subCommunicator(rc_comm), tile);
        return std::move(tile);
      },
      "recv_tile");
  return hpx::dataflow(ex, std::move(recv_bcast_f), mpi_task_chain());
}

// Local implementation of Lower Cholesky factorization.
template <class T>
void cholesky_L(Matrix<T, Device::CPU>& mat_a) {
  using hpx::threads::executors::pool_executor;
  using hpx::threads::thread_priority_high;
  using hpx::threads::thread_priority_default;

  // Set up executor on the default queue with high priority.
  pool_executor executor_hp("default", thread_priority_high);
  // Set up executor on the default queue with default priority.
  pool_executor executor_normal("default", thread_priority_default);

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
      // Choose queue priority
      auto trailing_matrix_executor = (j == k + 1) ? executor_hp : executor_normal;

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

// Distributed implementation of Lower Cholesky factorization.
template <class T>
void cholesky_L(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_a) {
  using hpx::threads::executors::pool_executor;
  using hpx::threads::thread_priority_high;
  using hpx::threads::thread_priority_default;
  using comm::internal::mpi_pool_exists;
  using ConstTile_t = matrix::Tile<const T, Device::CPU>;

  // Set up executor on the default queue with high priority.
  pool_executor executor_hp("default", thread_priority_high);
  // Set up executor on the default queue with default priority.
  pool_executor executor_normal("default", thread_priority_default);

  // Set up MPI executor
  comm::Index2D this_rank = grid.rank();
  auto executor_mpi = (mpi_pool_exists()) ? pool_executor("mpi", thread_priority_high) : executor_hp;
  common::Pipeline<comm::CommunicatorGrid> mpi_task_chain(std::move(grid));

  matrix::Distribution const& distr = mat_a.distribution();
  SizeType nrtile = mat_a.nrTiles().cols();

  for (SizeType k = 0; k < nrtile; ++k) {
    // Create a placeholder that will store the shared futures representing the panel
    std::unordered_map<SizeType, hpx::shared_future<ConstTile_t>> panel;

    GlobalTileIndex kk_idx(k, k);
    comm::Index2D kk_rank = distr.rankGlobalTile(kk_idx);

    // Broadcast the diagonal tile along the `k`-th column
    if (this_rank == kk_rank) {
      potrf_diag_tile(executor_hp, mat_a(kk_idx));
      panel[k] = mat_a.read(kk_idx);
      if (k != nrtile - 1)
        send_tile(executor_mpi, mpi_task_chain, Coord::Col, panel[k]);
    }
    else if (this_rank.col() == kk_rank.col()) {
      if (k != nrtile - 1)
        panel[k] = recv_tile<T>(executor_mpi, mpi_task_chain, Coord::Col, mat_a.tileSize(kk_idx),
                                kk_rank.row());
    }

    // Iterate over the k-th column
    std::vector<SizeType> bindices;
    for (SizeType i = k + 1; i < nrtile; ++i) {
      GlobalTileIndex ik_idx(i, k);
      comm::Index2D ik_rank = mat_a.rankGlobalTile(ik_idx);

      if (this_rank == ik_rank) {
        trsm_panel_tile(executor_hp, panel[k], mat_a(ik_idx));
        panel[i] = mat_a.read(ik_idx);
        send_tile(executor_mpi, mpi_task_chain, Coord::Row, panel[i]);
      }
      else if (this_rank.row() == ik_rank.row()) {
        panel[i] = recv_tile<T>(executor_mpi, mpi_task_chain, Coord::Row, mat_a.tileSize(ik_idx),
                                ik_rank.col());
      }
    }

    // Iterate over the trailing matrix
    for (SizeType j = k + 1; j < nrtile; ++j) {
      GlobalTileIndex jj_idx(j, j);
      comm::Index2D jj_rank = mat_a.rankGlobalTile(jj_idx);

      // Broadcast the jk-tile along the j-th column and update the jj-tile
      if (this_rank == jj_rank) {
        pool_executor trailing_matrix_executor = (j == k + 1) ? executor_hp : executor_normal;
        herk_trailing_diag_tile(trailing_matrix_executor, panel[j], mat_a(jj_idx));
        send_tile(executor_mpi, mpi_task_chain, Coord::Col, panel[j]);
      }
      else if (this_rank.col() == jj_rank.col()) {
        GlobalTileIndex jk_idx(j, k);
        panel[j] = recv_tile<T>(executor_mpi, mpi_task_chain, Coord::Col, mat_a.tileSize(jk_idx),
                                jj_rank.row());
      }

      for (SizeType i = j + 1; i < nrtile; ++i) {
        // Update the ij-tile using the ik-tile and jk-tile
        GlobalTileIndex ij_idx(i, j);
        comm::Index2D ij_rank = distr.rankGlobalTile(ij_idx);
        if (this_rank == ij_rank) {
          gemm_trailing_matrix_tile(executor_normal, panel[i], panel[j], mat_a(ij_idx));
        }
      }
    }
  }
}
}
}
}
