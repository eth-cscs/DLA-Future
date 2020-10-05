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
#include <hpx/lcos_fwd.hpp>
#include <hpx/util/annotated_function.hpp>

#include "dlaf/blas_tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/communication/pool.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace internal {
namespace mc {

// If the diagonal tile is on this node factorize it
// Cholesky decomposition on mat_a(k,k) r/w potrf (lapack operation)
template <class T>
void potrf_diag_tile(hpx::threads::executors::pool_executor executor_hp,
                     hpx::future<Tile<T, Device::CPU>> matrix_tile) {
  hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::potrf<T, Device::CPU>), blas::Uplo::Lower,
                std::move(matrix_tile));
}

template <class T>
void trsm_panel_tile(hpx::threads::executors::pool_executor executor_hp,
                     hpx::shared_future<Tile<const T, Device::CPU>> kk_tile,
                     hpx::future<Tile<T, Device::CPU>> matrix_tile) {
  hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), blas::Side::Right,
                blas::Uplo::Lower, blas::Op::ConjTrans, blas::Diag::NonUnit, 1.0, std::move(kk_tile),
                std::move(matrix_tile));
}

template <class T>
void herk_trailing_diag_tile(hpx::threads::executors::pool_executor trailing_matrix_executor,
                             hpx::shared_future<Tile<const T, Device::CPU>> panel_tile,
                             hpx::future<Tile<T, Device::CPU>> matrix_tile) {
  hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::herk<T, Device::CPU>),
                blas::Uplo::Lower, blas::Op::NoTrans, -1.0, panel_tile, 1.0, std::move(matrix_tile));
}

template <class T>
void gemm_trailing_matrix_tile(hpx::threads::executors::pool_executor trailing_matrix_executor,
                               hpx::shared_future<Tile<const T, Device::CPU>> panel_tile,
                               hpx::shared_future<Tile<const T, Device::CPU>> col_panel,
                               hpx::future<Tile<T, Device::CPU>> matrix_tile) {
  hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>),
                blas::Op::NoTrans, blas::Op::ConjTrans, -1.0, std::move(panel_tile),
                std::move(col_panel), 1.0, std::move(matrix_tile));
}

template <class T>
void send_tile(hpx::threads::executors::pool_executor executor_hp,
               common::Pipeline<comm::executor>& mpi_task_chain,
               hpx::shared_future<Tile<const T, Device::CPU>> tile) {
  using ConstTile_t = Tile<const T, Device::CPU>;
  using PromiseExec_t = common::PromiseGuard<comm::executor>;

  // Broadcast the (trailing) panel column-wise
  auto send_bcast_f = hpx::util::annotated_function(
      [](hpx::shared_future<ConstTile_t> fut_tile, hpx::future<PromiseExec_t> fpex) mutable {
        PromiseExec_t pex = fpex.get();
        comm::bcast(pex.ref(), pex.ref().comm().rank(), fut_tile.get());
      },
      "send_tile");
  hpx::dataflow(executor_hp, std::move(send_bcast_f), tile, mpi_task_chain());
}

template <class T>
hpx::shared_future<Tile<const T, Device::CPU>> recv_tile(
    hpx::threads::executors::pool_executor executor_hp, common::Pipeline<comm::executor>& mpi_task_chain,
    TileElementSize tile_size, int rank) {
  using ConstTile_t = Tile<const T, Device::CPU>;
  using PromiseExec_t = common::PromiseGuard<comm::executor>;
  using MemView_t = memory::MemoryView<T, Device::CPU>;
  using Tile_t = Tile<T, Device::CPU>;

  // Update the (trailing) panel column-wise
  auto recv_bcast_f = hpx::util::annotated_function(
      [rank, tile_size](hpx::future<PromiseExec_t> fpex) mutable {
        MemView_t mem_view(util::size_t::mul(tile_size.rows(), tile_size.cols()));
        Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());
        PromiseExec_t pex = fpex.get();
        return comm::bcast(pex.ref(), rank, tile)
            .then(hpx::launch::sync, [t = std::move(tile)](hpx::future<void>) mutable -> ConstTile_t {
              return std::move(t);
            });
      },
      "recv_tile");
  return hpx::future<ConstTile_t>(hpx::dataflow(executor_hp, std::move(recv_bcast_f), mpi_task_chain()));
}

/// Copy a tile
///
template <class T>
void copy_tile(TileElementSize ts, TileElementIndex in_begin_idx,
               Tile<const T, Device::CPU> const& in_tile, TileElementIndex out_begin_idx,
               Tile<T, Device::CPU> const& out_tile) {
  // TODO: `ts` must fit within in_tile and out_tile
  for (SizeType i = 0; i < ts.rows(); ++i) {
    for (SizeType j = 0; j < ts.cols(); ++j) {
      TileElementIndex out_idx(i + out_begin_idx.row(), j + out_begin_idx.col());
      TileElementIndex in_idx(i + in_begin_idx.row(), j + in_begin_idx.col());
      out_tile(out_idx) = in_tile(in_idx);
    }
  }
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

  using ConstTile_t = Tile<const T, Device::CPU>;

  // constexpr int max_pending_comms = 100;
  // constexpr int col_batch_size = 1;

  // Set up executor on the default queue with high priority.
  pool_executor executor_hp("default", thread_priority_high);
  // Set up executor on the default queue with default priority.
  pool_executor executor_normal("default", thread_priority_default);

  // Set up MPI executor pipelines
  comm::executor executor_mpi_col(grid.colCommunicator());
  comm::executor executor_mpi_row(grid.rowCommunicator());
  common::Pipeline<comm::executor> mpi_col_task_chain(std::move(executor_mpi_col));
  common::Pipeline<comm::executor> mpi_row_task_chain(std::move(executor_mpi_row));

  const matrix::Distribution& distr = mat_a.distribution();
  SizeType nrtile = mat_a.nrTiles().cols();
  // TileElementSize blk_size = mat_a.blockSize();

  if (nrtile == 0)
    return;

  SizeType localnrtile_rows = distr.localNrTiles().rows();
  comm::Index2D this_rank = grid.rank();

  for (SizeType k = 0; k < nrtile - 1; ++k) {
    // Create a placeholder that will store the shared futures representing the panel
    std::vector<hpx::shared_future<ConstTile_t>> panel(localnrtile_rows);

    GlobalTileIndex kk_idx(k, k);
    comm::Index2D kk_rank = distr.rankGlobalTile(kk_idx);
    hpx::shared_future<ConstTile_t> kk_tile;

    // Broadcast the diagonal tile along the `k`-th column
    if (this_rank == kk_rank) {
      potrf_diag_tile(executor_hp, mat_a(kk_idx));
      kk_tile = mat_a.read(kk_idx);
      send_tile(executor_hp, mpi_col_task_chain, kk_tile);
    }
    else if (this_rank.col() == kk_rank.col()) {
      kk_tile = recv_tile<T>(executor_hp, mpi_col_task_chain, mat_a.tileSize(kk_idx), kk_rank.row());
    }

    // Iterate over the k-th column
    for (SizeType i = k + 1; i < nrtile; ++i) {
      GlobalTileIndex ik_idx(i, k);
      comm::Index2D ik_rank = mat_a.rankGlobalTile(ik_idx);

      if (this_rank.row() != ik_rank.row())
        continue;

      // Update and broadcast each k-th column tile along it's row
      SizeType i_local = distr.localTileFromGlobalTile<Coord::Row>(i);
      if (this_rank.col() == ik_rank.col()) {
        trsm_panel_tile(executor_hp, kk_tile, mat_a(ik_idx));
        panel[i_local] = mat_a.read(ik_idx);
        send_tile(executor_hp, mpi_row_task_chain, mat_a.read(ik_idx));
      }
      else {
        panel[i_local] =
            recv_tile<T>(executor_hp, mpi_row_task_chain, mat_a.tileSize(ik_idx), kk_rank.col());
      }
    }

    // Iterate over the diagonal of the trailing matrix
    for (SizeType j = k + 1; j < nrtile; ++j) {
      pool_executor trailing_matrix_executor = (j == k + 1) ? executor_hp : executor_normal;
      // auto trailing_matrix_executor = (j_local == nextlocaltilek) ? executor_hp : executor_normal;
      GlobalTileIndex jj_idx(j, j);
      comm::Index2D jj_rank = mat_a.rankGlobalTile(jj_idx);

      // Broadcast the jk-tile along the j-th column and update the jj-tile
      hpx::shared_future<ConstTile_t> jk_tile;
      if (this_rank == jj_rank) {
        SizeType j_local = distr.localTileFromGlobalTile<Coord::Row>(j);
        jk_tile = panel[j_local];
        send_tile(executor_hp, mpi_col_task_chain, jk_tile);
        herk_trailing_diag_tile(trailing_matrix_executor, jk_tile, mat_a(jj_idx));
      }
      else if (this_rank.col() == jj_rank.col()) {
        GlobalTileIndex jk_idx(j, k);
        jk_tile = recv_tile<T>(executor_hp, mpi_col_task_chain, mat_a.tileSize(jk_idx), jj_rank.row());
      }

      // Iterate over the j-th column
      for (SizeType i = j + 1; i < nrtile; ++i) {
        // Update the ij-tile using the ik-tile and jk-tile
        GlobalTileIndex ij_idx(i, j);
        comm::Index2D ij_rank = distr.rankGlobalTile(ij_idx);
        if (this_rank == ij_rank) {
          SizeType i_local = distr.localTileFromGlobalTile<Coord::Row>(i);
          gemm_trailing_matrix_tile(trailing_matrix_executor, panel[i_local], jk_tile, mat_a(ij_idx));
        }
      }
    }
  }

  GlobalTileIndex last_idx(nrtile - 1, nrtile - 1);
  if (this_rank == distr.rankGlobalTile(last_idx)) {
    potrf_diag_tile(executor_hp, mat_a(GlobalTileIndex(last_idx)));
  }
}
}
}
}
