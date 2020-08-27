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

enum class transfer_op { load, offload };

template <transfer_op op>
struct swap_if_load_f {
  void operator()(SizeType&, SizeType&) noexcept {};  // noop
};

template <>
struct swap_if_load_f<transfer_op::load> {
  void operator()(SizeType& out_idx, SizeType& in_idx) noexcept {
    std::swap(out_idx, in_idx);
  };
};

/// Load a tile from the panel into the matrix or offload a tile from the panel into the matrix.
///
/// @p matrix_tile_index the index of the matrix tile within the panel
template <class T, transfer_op op>
void transfer_tile(hpx::threads::executors::pool_executor executor_hp,
                   hpx::shared_future<Tile<const T, Device::CPU>> in_tile,
                   hpx::future<Tile<T, Device::CPU>> out_tile, int tile_idx) {
  using ConstTile_t = Tile<const T, Device::CPU>;
  using Tile_t = Tile<T, Device::CPU>;
  auto load_f = [tile_idx](hpx::shared_future<ConstTile_t> in_ftile, hpx::future<Tile_t> out_ftile) {
    ConstTile_t in_tile = in_ftile.get();
    Tile_t out_tile = out_ftile.get();
    TileElementSize out_ts = out_tile.size();
    SizeType pt_offset = tile_idx * out_ts.rows();
    for (SizeType mt_i = 0; mt_i < out_ts.rows(); ++mt_i) {
      for (SizeType mt_j = 0; mt_j < out_ts.cols(); ++mt_j) {
        TileElementIndex out_idx(mt_i, mt_j);             // matrix tile
        TileElementIndex in_idx(pt_offset + mt_i, mt_j);  // panel tile
        swap_if_load_f<op>(out_idx, in_idx);
        out_tile(out_idx) = in_tile(in_idx);
      }
    }
  };
  hpx::dataflow(executor_hp, std::move(load_f), std::move(in_tile), std::move(out_tile));
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

      // Update trailing matrix: diagonal element mat_a(j,j, reading mat_a.read(j,k), using herk (blas operation)
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

  auto localnrtile_rows = distr.localNrTiles().rows();
  auto localnrtile_cols = distr.localNrTiles().cols();

  auto this_rank = grid.rank();

  for (SizeType k = 0; k < nrtile; ++k) {
    // Create a placeholder that will store the shared futures representing the panel
    std::vector<hpx::shared_future<ConstTile_t>> panel(localnrtile_rows);

    auto kk_idx = GlobalTileIndex(k, k);
    auto kk_tile_rank = distr.rankGlobalTile(kk_idx);

    if (this_rank.col() == kk_tile_rank.col()) {
      // if the process is on the column of the diagonal tile and the panel
      hpx::shared_future<ConstTile_t> kk_tile;

      if (this_rank.row() == kk_tile_rank.row()) {
        // if the process has the diagonal tile, factorize and send it
        potrf_diag_tile(executor_hp, mat_a(kk_idx));
        kk_tile = mat_a.read(kk_idx);
        send_tile(executor_hp, mpi_col_task_chain, kk_tile);
      }
      else {
        // receive the diagonal tile in processes holding panel tiles
        kk_tile =
            recv_tile<T>(executor_hp, mpi_col_task_chain, mat_a.tileSize(kk_idx), kk_tile_rank.row());
      }

      auto k_local_col = distr.localTileFromGlobalTile<Coord::Col>(k);
      for (SizeType i_local = distr.nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
           i_local < localnrtile_rows; ++i_local) {
        LocalTileIndex ik_local_idx(i_local, k_local_col);
        trsm_panel_tile(executor_hp, kk_tile, mat_a(ik_local_idx));
        panel[i_local] = mat_a.read(ik_local_idx);
        send_tile(executor_hp, mpi_row_task_chain, mat_a.read(ik_local_idx));
      }
    }
    else {
      for (SizeType i_local = distr.nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
           i_local < localnrtile_rows; ++i_local) {
        auto i = distr.globalTileFromLocalTile<Coord::Row>(i_local);
        panel[i_local] = recv_tile<T>(executor_hp, mpi_row_task_chain,
                                      mat_a.tileSize(GlobalTileIndex(i, k)), kk_tile_rank.col());
      }
    }

    auto nextlocaltilek = distr.nextLocalTileFromGlobalTile<Coord::Col>(k + 1);
    for (SizeType j_local = nextlocaltilek; j_local < localnrtile_cols; ++j_local) {
      auto j = distr.globalTileFromLocalTile<Coord::Col>(j_local);

      // Choose "high priority" for first tile of the trailing matrix
      // Version 1: only the global k + 1 column is "high priority"
      auto trailing_matrix_executor = (j == k + 1) ? executor_hp : executor_normal;

      // TODO: benchmark to find faster
      // Version 2: the next column of each rank is "high priority"
      //        auto trailing_matrix_executor = (j_local == nextlocaltilek) ? executor_hp :
      //        executor_normal;

      hpx::shared_future<ConstTile_t> col_panel;

      auto j_rank_row = distr.rankGlobalTile<Coord::Row>(j);

      if (this_rank.row() == j_rank_row) {
        auto i_local = distr.localTileFromGlobalTile<Coord::Row>(j);
        col_panel = panel[i_local];
        send_tile(executor_hp, mpi_col_task_chain, col_panel);

        // Compute first tile of the column of the trailing matrix: diagonal element mat_a(j,j),
        // reading mat_a.read(j,k), using herk (blas operation)
        herk_trailing_diag_tile(trailing_matrix_executor, col_panel, mat_a(GlobalTileIndex{j, j}));
      }
      else {
        col_panel = recv_tile<T>(executor_hp, mpi_col_task_chain, mat_a.tileSize(GlobalTileIndex(j, k)),
                                 j_rank_row);
      }

      for (SizeType i_local = distr.nextLocalTileFromGlobalTile<Coord::Row>(j + 1);
           i_local < localnrtile_rows; ++i_local) {
        // Update remaining trailing matrix mat_a(i,j), reading mat_a.read(i,k) and mat_a.read(j,k),
        // using gemm (blas operation)
        gemm_trailing_matrix_tile(trailing_matrix_executor, panel[i_local], col_panel,
                                  mat_a(LocalTileIndex{i_local, j_local}));
      }
    }
  }
}
}
}
}
