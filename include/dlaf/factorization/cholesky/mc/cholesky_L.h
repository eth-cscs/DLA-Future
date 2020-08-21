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

// Local implementation of Lower Cholesky factorization.
template <class T>
void cholesky_L(Matrix<T, Device::CPU>& mat_a) {
  constexpr auto NonUnit = blas::Diag::NonUnit;
  constexpr auto ConjTrans = blas::Op::ConjTrans;
  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto Right = blas::Side::Right;
  constexpr auto Lower = blas::Uplo::Lower;

  using hpx::util::unwrapping;
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

    hpx::dataflow(executor_hp, unwrapping(tile::potrf<T, Device::CPU>), Lower, std::move(mat_a(kk)));

    for (SizeType i = k + 1; i < nrtile; ++i) {
      // Update panel mat_a(i,k) with trsm (blas operation), using data mat_a.read(k,k)
      hpx::dataflow(executor_hp, unwrapping(tile::trsm<T, Device::CPU>), Right, Lower, ConjTrans,
                    NonUnit, 1.0, mat_a.read(kk), std::move(mat_a(LocalTileIndex{i, k})));
    }

    for (SizeType j = k + 1; j < nrtile; ++j) {
      // Choose queue priority
      auto trailing_matrix_executor = (j == k + 1) ? executor_hp : executor_normal;

      // Update trailing matrix: diagonal element mat_a(j,j, reading mat_a.read(j,k), using herk (blas operation)
      hpx::dataflow(trailing_matrix_executor, unwrapping(tile::herk<T, Device::CPU>), Lower, NoTrans,
                    -1.0, mat_a.read(LocalTileIndex{j, k}), 1.0, std::move(mat_a(LocalTileIndex{j, j})));

      for (SizeType i = j + 1; i < nrtile; ++i) {
        // Update remaining trailing matrix mat_a(i,j), reading mat_a.read(i,k) and mat_a.read(j,k),
        // using gemm (blas operation)
        hpx::dataflow(trailing_matrix_executor, unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
                      ConjTrans, -1.0, mat_a.read(LocalTileIndex{i, k}),
                      mat_a.read(LocalTileIndex{j, k}), 1.0, std::move(mat_a(LocalTileIndex{i, j})));
      }
    }
  }
}

template <class T>
hpx::shared_future<Tile<const T, Device::CPU>> factorize_and_send_diag_tile(
    hpx::threads::executors::pool_executor executor_hp, Matrix<T, Device::CPU>& mat_a,
    common::Pipeline<comm::executor>& mpi_col_task_chain, const matrix::Distribution& distr,
    GlobalTileIndex kk_idx) {
  using ConstTile_t = Tile<const T, Device::CPU>;
  using PromiseExec_t = common::PromiseGuard<comm::executor>;

  LocalTileIndex kk = distr.localTileIndex(kk_idx);

  // If the diagonal tile is on this node factorize it
  // Cholesky decomposition on mat_a(k,k) r/w potrf (lapack operation)
  hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::potrf<T, Device::CPU>), blas::Uplo::Lower,
                std::move(mat_a(kk)));

  // Broadcast the panel column-wise
  auto send_bcast_f = hpx::util::annotated_function(
      [](hpx::shared_future<ConstTile_t> ftile, hpx::future<PromiseExec_t> fpex) mutable {
        PromiseExec_t pex = fpex.get();
        comm::bcast(pex.ref(), pex.ref().comm().rank(), ftile.get());
      },
      "send_diag_tile");
  hpx::dataflow(executor_hp, std::move(send_bcast_f), mat_a.read(kk), mpi_col_task_chain());

  return mat_a.read(kk);
}

// If this process holds tiles in the panel, receive the diagonal tile
template <class T>
hpx::shared_future<Tile<const T, Device::CPU>> recv_diag_tile(
    hpx::threads::executors::pool_executor executor_hp, Matrix<T, Device::CPU>& mat_a,
    common::Pipeline<comm::executor>& mpi_col_task_chain, comm::Index2D kk_tile_rank,
    GlobalTileIndex kk_idx) {
  using Tile_t = Tile<T, Device::CPU>;
  using ConstTile_t = Tile<const T, Device::CPU>;
  using PromiseExec_t = common::PromiseGuard<comm::executor>;
  using MemView_t = memory::MemoryView<T, Device::CPU>;

  // Receive the diagonal tile
  auto recv_bcast_f = hpx::util::annotated_function(
      [rank = kk_tile_rank.row(),
       tile_size = mat_a.tileSize(kk_idx)](hpx::future<PromiseExec_t> fpex) mutable {
        MemView_t mem_view(util::size_t::mul(tile_size.rows(), tile_size.cols()));
        Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());
        PromiseExec_t pex = fpex.get();
        return comm::bcast(pex.ref(), rank, tile)
            .then(hpx::launch::sync, [t = std::move(tile)](hpx::future<void>) mutable -> ConstTile_t {
              return std::move(t);
            });
      },
      "recv_diag_tile");
  return hpx::future<ConstTile_t>(
      hpx::dataflow(executor_hp, std::move(recv_bcast_f), mpi_col_task_chain()));
}

template <class T>
void solve_and_send_panel(hpx::threads::executors::pool_executor executor_hp,
                          Matrix<T, Device::CPU>& mat_a,
                          common::Pipeline<comm::executor>& mpi_row_task_chain,
                          const matrix::Distribution& distr, SizeType k,
                          hpx::shared_future<Tile<const T, Device::CPU>> kk_tile,
                          std::vector<hpx::shared_future<Tile<const T, Device::CPU>>>& panel) {
  using ConstTile_t = Tile<const T, Device::CPU>;
  using PromiseExec_t = common::PromiseGuard<comm::executor>;

  auto k_local_col = distr.localTileFromGlobalTile<Coord::Col>(k);
  for (SizeType i_local = distr.nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
       i_local < distr.localNrTiles().rows(); ++i_local) {
    LocalTileIndex ik_local_idx(i_local, k_local_col);
    // Update panel mat_a(i,k) with trsm (blas operation), using data mat_a.read(k,k)
    hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), blas::Side::Right,
                  blas::Uplo::Lower, blas::Op::ConjTrans, blas::Diag::NonUnit, 1.0, kk_tile,
                  std::move(mat_a(ik_local_idx)));

    // Avoid useless communications if one-row communicator grid
    // Broadcast the panel row-wise
    auto send_bcast_f = hpx::util::annotated_function(
        [](hpx::shared_future<ConstTile_t> sftile, hpx::future<PromiseExec_t> fpex) mutable {
          PromiseExec_t pex = fpex.get();
          comm::bcast(pex.ref(), pex.ref().comm().rank(), sftile.get());
        },
        "send_panel");
    hpx::dataflow(executor_hp, std::move(send_bcast_f), mat_a.read(ik_local_idx), mpi_row_task_chain());

    panel[i_local] = mat_a.read(ik_local_idx);
  }
}

template <class T>
void recv_panel(hpx::threads::executors::pool_executor executor_hp, Matrix<T, Device::CPU>& mat_a,
                common::Pipeline<comm::executor>& mpi_row_task_chain, const matrix::Distribution& distr,
                SizeType k, comm::Index2D kk_tile_rank,
                std::vector<hpx::shared_future<Tile<const T, Device::CPU>>>& panel) {
  using ConstTile_t = Tile<const T, Device::CPU>;
  using PromiseExec_t = common::PromiseGuard<comm::executor>;
  using MemView_t = memory::MemoryView<T, Device::CPU>;
  using Tile_t = Tile<T, Device::CPU>;

  for (SizeType i_local = distr.nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
       i_local < distr.localNrTiles().rows(); ++i_local) {
    auto i = distr.globalTileFromLocalTile<Coord::Row>(i_local);

    // Receiving the panel
    auto recv_bcast_f = hpx::util::annotated_function(
        [rank = kk_tile_rank.col(),
         tile_size = mat_a.tileSize(GlobalTileIndex(i, k))](hpx::future<PromiseExec_t> fpex) mutable {
          MemView_t mem_view(util::size_t::mul(tile_size.rows(), tile_size.cols()));
          Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());
          PromiseExec_t pex = fpex.get();
          return comm::bcast(pex.ref(), rank, tile)
              .then(hpx::launch::sync, [t = std::move(tile)](hpx::future<void>) mutable -> ConstTile_t {
                return std::move(t);
              });
        },
        "recv_panel");
    panel[i_local] = hpx::future<ConstTile_t>(
        hpx::dataflow(executor_hp, std::move(recv_bcast_f), mpi_row_task_chain()));
  }
}

template <class T>
void send_panel_tile(hpx::threads::executors::pool_executor executor_hp,
                     common::Pipeline<comm::executor>& mpi_col_task_chain,
                     hpx::shared_future<Tile<const T, Device::CPU>> panel_tile) {
  using ConstTile_t = Tile<const T, Device::CPU>;
  using PromiseExec_t = common::PromiseGuard<comm::executor>;

  // Broadcast the (trailing) panel column-wise
  auto send_bcast_f = hpx::util::annotated_function(
      [](hpx::shared_future<ConstTile_t> fut_tile, hpx::future<PromiseExec_t> fpex) mutable {
        PromiseExec_t pex = fpex.get();
        comm::bcast(pex.ref(), pex.ref().comm().rank(), fut_tile.get());
      },
      "send_trailing_panel");
  hpx::dataflow(executor_hp, std::move(send_bcast_f), panel_tile, mpi_col_task_chain());
}

// Compute first tile of the column of the trailing matrix: diagonal element mat_a(j,j),
// reading mat_a.read(j,k), using herk (blas operation)
template <class T>
void herk_trailing_diag_tile(hpx::threads::executors::pool_executor trailing_matrix_executor,
                             hpx::future<Tile<T, Device::CPU>> matrix_tile,
                             hpx::shared_future<Tile<const T, Device::CPU>> panel_tile) {
  hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::herk<T, Device::CPU>),
                blas::Uplo::Lower, blas::Op::NoTrans, -1.0, panel_tile, 1.0, std::move(matrix_tile));
}

template <class T>
hpx::shared_future<Tile<const T, Device::CPU>> recv_panel_tile_in_trailing_panel(
    hpx::threads::executors::pool_executor executor_hp, Matrix<T, Device::CPU>& mat_a,
    common::Pipeline<comm::executor>& mpi_col_task_chain, int j_rank_row, SizeType j, SizeType k) {
  using ConstTile_t = Tile<const T, Device::CPU>;
  using PromiseExec_t = common::PromiseGuard<comm::executor>;
  using MemView_t = memory::MemoryView<T, Device::CPU>;
  using Tile_t = Tile<T, Device::CPU>;

  // Update the (trailing) panel column-wise
  auto recv_bcast_f = hpx::util::annotated_function(
      [rank = j_rank_row,
       tile_size = mat_a.tileSize(GlobalTileIndex(j, k))](hpx::future<PromiseExec_t> fpex) mutable {
        MemView_t mem_view(util::size_t::mul(tile_size.rows(), tile_size.cols()));
        Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());
        PromiseExec_t pex = fpex.get();
        return comm::bcast(pex.ref(), rank, tile)
            .then(hpx::launch::sync, [t = std::move(tile)](hpx::future<void>) mutable -> ConstTile_t {
              return std::move(t);
            });
      },
      "recv_trailing_panel");
  return hpx::future<ConstTile_t>(
      hpx::dataflow(executor_hp, std::move(recv_bcast_f), mpi_col_task_chain()));
}

// Distributed implementation of Lower Cholesky factorization.
template <class T>
void cholesky_L(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_a) {
  using hpx::threads::executors::pool_executor;
  using hpx::threads::thread_priority_high;
  using hpx::threads::thread_priority_default;
  using comm::internal::mpi_pool_exists;
  using common::internal::vector;

  using ConstTile_t = Tile<const T, Device::CPU>;

  // constexpr int max_pending_comms = 100;

  // Set up executor on the default queue with high priority.
  pool_executor executor_hp("default", thread_priority_high);
  // Set up executor on the default queue with default priority.
  pool_executor executor_normal("default", thread_priority_default);
  // Set up MPI executor
  std::string mpi_pool_name = (mpi_pool_exists()) ? "mpi" : "default";
  comm::executor executor_mpi_col(mpi_pool_name, grid.colCommunicator());
  comm::executor executor_mpi_row(mpi_pool_name, grid.rowCommunicator());

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
        kk_tile = factorize_and_send_diag_tile(executor_hp, mat_a, mpi_col_task_chain, distr, kk_idx);
      }
      else {
        // receive the diagonal tile in processes holding panel tiles
        kk_tile = recv_diag_tile(executor_hp, mat_a, mpi_col_task_chain, kk_tile_rank, kk_idx);
      }

      solve_and_send_panel(executor_hp, mat_a, mpi_row_task_chain, distr, k, std::move(kk_tile), panel);
    }
    else {
      recv_panel(executor_hp, mat_a, mpi_row_task_chain, distr, k, kk_tile_rank, panel);
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
        send_panel_tile(executor_hp, mpi_col_task_chain, col_panel);
        herk_trailing_diag_tile(trailing_matrix_executor, mat_a(LocalTileIndex{i_local, j_local}),
                                col_panel);
      }
      else {
        col_panel =
            recv_panel_tile_in_trailing_panel(executor_hp, mat_a, mpi_col_task_chain, j_rank_row, j, k);
      }

      for (SizeType i_local = distr.nextLocalTileFromGlobalTile<Coord::Row>(j + 1);
           i_local < localnrtile_rows; ++i_local) {
        // Update remaining trailing matrix mat_a(i,j), reading mat_a.read(i,k) and mat_a.read(j,k),
        // using gemm (blas operation)
        hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>),
                      blas::Op::NoTrans, blas::Op::ConjTrans, -1.0, panel[i_local], col_panel, 1.0,
                      std::move(mat_a(LocalTileIndex{i_local, j_local})));
      }
    }
  }
}
}
}
}
