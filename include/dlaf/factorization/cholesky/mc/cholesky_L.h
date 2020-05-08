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

#include <unistd.h>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/util/unwrap.hpp>

#include "dlaf/blas_tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/functions_sync.h"
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

  using std::move;

  using hpx::threads::executors::pool_executor;
  using hpx::threads::thread_priority_high;
  using hpx::threads::thread_priority_default;
  using hpx::util::unwrapping;
  using hpx::dataflow;

  // Set up executor on the default queue with high priority.
  pool_executor executor_hp("default", thread_priority_high);
  // Set up executor on the default queue with default priority.
  pool_executor executor_normal("default", thread_priority_default);

  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().cols();

  for (SizeType k = 0; k < nrtile; ++k) {
    // Cholesky decomposition on mat_a(k,k) r/w potrf (lapack operation)
    auto kk = LocalTileIndex{k, k};

    auto potrf_f = unwrapping([Lower](auto&& tile) { tile::potrf<T, Device::CPU>(Lower, tile); });
    dataflow(executor_hp, move(potrf_f), move(mat_a(kk)));

    for (SizeType i = k + 1; i < nrtile; ++i) {
      // Update panel mat_a(i,k) with trsm (blas operation), using data mat_a.read(k,k)
      auto trsm_f = unwrapping([Right, Lower, ConjTrans, NonUnit](auto&& kk_tile, auto&& ik_tile) {
        tile::trsm<T, Device::CPU>(Right, Lower, ConjTrans, NonUnit, 1.0, kk_tile, ik_tile);
      });
      dataflow(executor_hp, move(trsm_f), mat_a.read(kk), move(mat_a(LocalTileIndex{i, k})));
    }

    for (SizeType j = k + 1; j < nrtile; ++j) {
      // Choose queue priority
      auto trailing_matrix_executor = (j == k + 1) ? executor_hp : executor_normal;

      // Update trailing matrix: diagonal element mat_a(j,j, reading mat_a.read(j,k), using herk (blas operation)
      auto herk_f = unwrapping([Lower, NoTrans](auto&& jk_tile, auto&& jj_tile) {
        tile::herk<T, Device::CPU>(Lower, NoTrans, -1.0, jk_tile, 1.0, jj_tile);
      });
      dataflow(trailing_matrix_executor, move(herk_f), mat_a.read(LocalTileIndex{j, k}),
               move(mat_a(LocalTileIndex{j, j})));

      for (SizeType i = j + 1; i < nrtile; ++i) {
        // Update remaining trailing matrix mat_a(i,j), reading mat_a.read(i,k) and mat_a.read(j,k),
        // using gemm (blas operation)
        auto gemm_f = unwrapping([NoTrans, ConjTrans](auto&& ik_tile, auto&& jk_tile, auto&& ij_tile) {
          tile::gemm<T, Device::CPU>(NoTrans, ConjTrans, -1.0, ik_tile, jk_tile, 1.0, ij_tile);
        });
        dataflow(trailing_matrix_executor, move(gemm_f), mat_a.read(LocalTileIndex{i, k}),
                 mat_a.read(LocalTileIndex{j, k}), move(mat_a(LocalTileIndex{i, j})));
      }
    }
  }
}

// Distributed implementation of Lower Cholesky factorization.
template <class T>
void cholesky_L(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_a) {
  using std::move;

  using hpx::util::unwrapping;
  using hpx::dataflow;
  using hpx::threads::executors::pool_executor;
  using hpx::threads::thread_priority_high;
  using hpx::threads::thread_priority_default;

  using common::internal::vector;
  using comm::mpi_pool_exists;

  using TileType = Tile<T, Device::CPU>;
  using ConstTileType = Tile<const T, Device::CPU>;
  using MemViewType = memory::MemoryView<T, Device::CPU>;

  constexpr auto NonUnit = blas::Diag::NonUnit;
  constexpr auto ConjTrans = blas::Op::ConjTrans;
  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto Right = blas::Side::Right;
  constexpr auto Lower = blas::Uplo::Lower;

  // Set up executor on the default queue with high priority.
  pool_executor executor_hp("default", thread_priority_high);
  // Set up executor on the default queue with default priority.
  pool_executor executor_normal("default", thread_priority_default);

  auto executor_mpi = (mpi_pool_exists()) ? pool_executor("mpi", thread_priority_high) : executor_hp;

  auto col_comm_size = grid.colCommunicator().size();
  auto row_comm_size = grid.rowCommunicator().size();

  const matrix::Distribution& distr = mat_a.distribution();

  SizeType nrtile = mat_a.nrTiles().cols();

  auto localnrtile_rows = distr.localNrTiles().rows();
  auto localnrtile_cols = distr.localNrTiles().cols();

  common::Pipeline<comm::CommunicatorGrid> serial_comm(move(grid));

  for (SizeType k = 0; k < nrtile; ++k) {
    // Create a placeholder that will store the shared futures representing the panel
    vector<hpx::shared_future<ConstTileType>> panel(distr.localNrTiles().rows());

    auto k_rank_row = distr.rankGlobalTile<Coord::Row>(k);
    auto k_rank_col = distr.rankGlobalTile<Coord::Col>(k);

    if (mat_a.rankIndex().col() == k_rank_col) {
      auto k_local_col = distr.localTileFromGlobalTile<Coord::Col>(k);

      hpx::shared_future<ConstTileType> kk_tile;

      if (mat_a.rankIndex().row() == k_rank_row) {
        auto k_local_row = distr.localTileFromGlobalTile<Coord::Row>(k);

        auto kk = LocalTileIndex{k_local_row, k_local_col};

        // If the diagonal tile is on this node factorize it
        // Cholesky decomposition on mat_a(k,k) r/w potrf (lapack operation)
        auto potrf_f =
            unwrapping([Lower](auto&& kk_tile) { tile::potrf<T, Device::CPU>(Lower, kk_tile); });
        dataflow(executor_hp, move(potrf_f), move(mat_a(kk)));

        // Avoid useless communication if one-column communicator and if on the last column
        if (col_comm_size > 1 && k != (mat_a.nrTiles().cols() - 1)) {
          // Broadcast the panel column-wise
          auto send_f = unwrapping([](auto&& tile, auto&& comm_wrapper) {
            comm::sync::broadcast::send(comm_wrapper().colCommunicator(), tile);
          });
          dataflow(move(send_f), mat_a.read(kk), serial_comm());
        }

        kk_tile = mat_a.read(kk);
      }
      else {
        // Avoid useless communications if one-column communicator and if on the last column
        if (col_comm_size > 1 && k != (mat_a.nrTiles().cols() - 1)) {
          // Receive the diagonal tile
          auto tile_size = mat_a.tileSize(GlobalTileIndex(k, k));
          auto recv_f = unwrapping([k_rank_row, tile_size](auto&& comm_wrapper) -> ConstTileType {
            MemViewType mem_view(util::size_t::mul(tile_size.rows(), tile_size.cols()));
            Tile<T, Device::CPU> tile(tile_size, move(mem_view), tile_size.rows());
            comm::sync::broadcast::receive_from(k_rank_row, comm_wrapper().colCommunicator(), tile);
            return move(tile);
          });
          kk_tile = dataflow(executor_mpi, move(recv_f), serial_comm());
        }
      }

      for (SizeType i_local = distr.nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
           i_local < localnrtile_rows; ++i_local) {
        // Update panel mat_a(i,k) with trsm (blas operation), using data mat_a.read(k,k)
        auto trsm_f = unwrapping([Right, Lower, ConjTrans, NonUnit](auto&& a, auto&& b) {
          tile::trsm<T, Device::CPU>(Right, Lower, ConjTrans, NonUnit, 1.0, a, b);
        });
        dataflow(executor_hp, move(trsm_f), kk_tile, move(mat_a(LocalTileIndex{i_local, k_local_col})));

        // Avoid useless communications if one-row communicator grid
        if (row_comm_size > 1) {
          // Broadcast the panel row-wise
          auto send_f = unwrapping([](auto&& tile, auto&& comm_wrapper) {
            comm::sync::broadcast::send(comm_wrapper().rowCommunicator(), tile);
          });
          dataflow(executor_mpi, move(send_f), mat_a.read(LocalTileIndex{i_local, k_local_col}),
                   serial_comm());
        }
        panel[i_local] = mat_a.read(LocalTileIndex{i_local, k_local_col});
      }
    }
    else {
      for (SizeType i_local = distr.nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
           i_local < localnrtile_rows; ++i_local) {
        auto i = distr.globalTileFromLocalTile<Coord::Row>(i_local);

        // Avoid useless communications if one-row communicator grid
        if (row_comm_size > 1) {
          // Receiving the panel
          auto tile_size = mat_a.tileSize(GlobalTileIndex(i, k));
          auto recv_panel_f = unwrapping([k_rank_col, tile_size](auto&& comm_wrapper) -> ConstTileType {
            MemViewType mem_view(util::size_t::mul(tile_size.rows(), tile_size.cols()));
            TileType tile(tile_size, move(mem_view), tile_size.rows());
            comm::sync::broadcast::receive_from(k_rank_col, comm_wrapper().rowCommunicator(), tile);
            return move(tile);
          });
          panel[i_local] = dataflow(executor_mpi, move(recv_panel_f), serial_comm());
        }
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

      hpx::shared_future<ConstTileType> col_panel;

      auto j_rank_row = distr.rankGlobalTile<Coord::Row>(j);

      if (mat_a.rankIndex().row() == j_rank_row) {
        auto i_local = distr.localTileFromGlobalTile<Coord::Row>(j);

        // Avoid useless communications if one-row communicator grid and if on the last panel
        if (col_comm_size > 1 && j != (mat_a.nrTiles().cols() - 1)) {
          // Broadcast the (trailing) panel column-wise
          auto send_f = unwrapping([](auto&& tile, auto&& comm_wrapper) {
            comm::sync::broadcast::send(comm_wrapper().colCommunicator(), tile);
          });
          dataflow(executor_mpi, move(send_f), panel[i_local], serial_comm());
        }

        // Check if the diagonal tile of the trailing matrix is on this node and
        // compute first tile of the column of the trailing matrix: diagonal element mat_a(j,j), reading
        // mat_a.read(j,k), using herk (blas operation)
        auto herk_f = unwrapping([Lower, NoTrans](auto&& panel_tile, auto&& a_tile) {
          tile::herk<T, Device::CPU>(Lower, NoTrans, -1.0, panel_tile, 1.0, a_tile);
        });
        dataflow(trailing_matrix_executor, move(herk_f), panel[i_local],
                 mat_a(LocalTileIndex{i_local, j_local}));

        col_panel = panel[i_local];
      }
      else {
        // Avoid useless communications if one-row communicator grid and if on the last panel
        if (col_comm_size > 1 && j != (mat_a.nrTiles().cols() - 1)) {
          // Update the (trailing) panel column-wise
          auto tile_size = mat_a.tileSize(GlobalTileIndex(j, k));
          auto recv_f = unwrapping([j_rank_row, tile_size](auto&& comm_wrapper) -> ConstTileType {
            MemViewType mem_view(util::size_t::mul(tile_size.rows(), tile_size.cols()));
            TileType tile(tile_size, move(mem_view), tile_size.rows());
            comm::sync::broadcast::receive_from(j_rank_row, comm_wrapper().colCommunicator(), tile);
            return move(tile);
          });
          col_panel = dataflow(executor_mpi, move(recv_f), serial_comm());
        }
      }

      for (SizeType i_local = distr.nextLocalTileFromGlobalTile<Coord::Row>(j + 1);
           i_local < localnrtile_rows; ++i_local) {
        // Update remaining trailing matrix mat_a(i,j), reading mat_a.read(i,k) and mat_a.read(j,k),
        // using gemm (blas operation)
        auto gemm_f =
            unwrapping([NoTrans, ConjTrans](auto&& iloc_panel, auto&& col_panel, auto&& a_tile) {
              tile::gemm<T, Device::CPU>(NoTrans, ConjTrans, -1.0, iloc_panel, col_panel, 1.0, a_tile);
            });
        dataflow(trailing_matrix_executor, move(gemm_f), panel[i_local], col_panel,
                 move(mat_a(LocalTileIndex{i_local, j_local})));
      }
    }
  }
}
}
}
}
