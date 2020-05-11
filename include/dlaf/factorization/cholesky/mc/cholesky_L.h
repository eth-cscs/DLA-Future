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

    hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::potrf<T, Device::CPU>), Lower,
                  std::move(mat_a(kk)));

    for (SizeType i = k + 1; i < nrtile; ++i) {
      // Update panel mat_a(i,k) with trsm (blas operation), using data mat_a.read(k,k)
      hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), Right, Lower,
                    ConjTrans, NonUnit, 1.0, mat_a.read(kk), std::move(mat_a(LocalTileIndex{i, k})));
    }

    for (SizeType j = k + 1; j < nrtile; ++j) {
      // Choose queue priority
      auto trailing_matrix_executor = (j == k + 1) ? executor_hp : executor_normal;

      // Update trailing matrix: diagonal element mat_a(j,j, reading mat_a.read(j,k), using herk (blas operation)
      hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::herk<T, Device::CPU>), Lower,
                    NoTrans, -1.0, mat_a.read(LocalTileIndex{j, k}), 1.0,
                    std::move(mat_a(LocalTileIndex{j, j})));

      for (SizeType i = j + 1; i < nrtile; ++i) {
        // Update remaining trailing matrix mat_a(i,j), reading mat_a.read(i,k) and mat_a.read(j,k),
        // using gemm (blas operation)
        hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>),
                      NoTrans, ConjTrans, -1.0, mat_a.read(LocalTileIndex{i, k}),
                      mat_a.read(LocalTileIndex{j, k}), 1.0, std::move(mat_a(LocalTileIndex{i, j})));
      }
    }
  }
}

// Distributed implementation of Lower Cholesky factorization.
template <class T>
void cholesky_L(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_a) {
  using common::internal::vector;

  using hpx::threads::executors::pool_executor;
  using hpx::threads::thread_priority_high;
  using hpx::threads::thread_priority_default;

  using comm::internal::mpi_pool_exists;

  constexpr auto NonUnit = blas::Diag::NonUnit;
  constexpr auto ConjTrans = blas::Op::ConjTrans;
  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto Right = blas::Side::Right;
  constexpr auto Lower = blas::Uplo::Lower;

  // Set up executor on the default queue with high priority.
  pool_executor executor_hp("default", thread_priority_high);
  // Set up executor on the default queue with default priority.
  pool_executor executor_normal("default", thread_priority_default);
  // Set up MPI executor
  auto executor_mpi = (mpi_pool_exists()) ? pool_executor("mpi", thread_priority_high) : executor_hp;

  auto col_comm_size = grid.colCommunicator().size();
  auto row_comm_size = grid.rowCommunicator().size();

  const matrix::Distribution& distr = mat_a.distribution();

  SizeType nrtile = mat_a.nrTiles().cols();

  auto localnrtile_rows = distr.localNrTiles().rows();
  auto localnrtile_cols = distr.localNrTiles().cols();

  common::Pipeline<comm::CommunicatorGrid> serial_comm(std::move(grid));

  for (SizeType k = 0; k < nrtile; ++k) {
    // Create a placeholder that will store the shared futures representing the panel
    vector<hpx::shared_future<Tile<const T, Device::CPU>>> panel(distr.localNrTiles().rows());

    auto k_rank_row = distr.rankGlobalTile<Coord::Row>(k);
    auto k_rank_col = distr.rankGlobalTile<Coord::Col>(k);

    if (mat_a.rankIndex().col() == k_rank_col) {
      auto k_local_col = distr.localTileFromGlobalTile<Coord::Col>(k);

      hpx::shared_future<Tile<const T, Device::CPU>> kk_tile;

      if (mat_a.rankIndex().row() == k_rank_row) {
        auto k_local_row = distr.localTileFromGlobalTile<Coord::Row>(k);

        auto kk = LocalTileIndex{k_local_row, k_local_col};

        // If the diagonal tile is on this node factorize it
        // Cholesky decomposition on mat_a(k,k) r/w potrf (lapack operation)
        hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::potrf<T, Device::CPU>), Lower,
                      std::move(mat_a(kk)));

        // Avoid useless communication if one-column communicator and if on the last column
        if (col_comm_size > 1 && k != (mat_a.nrTiles().cols() - 1)) {
          // Broadcast the panel column-wise
          hpx::dataflow(hpx::util::unwrapping([](auto&& tile, auto&& comm_wrapper) {
                          comm::sync::broadcast::send(comm_wrapper().colCommunicator(), tile);
                        }),
                        mat_a.read(kk), serial_comm());
        }

        kk_tile = mat_a.read(kk);
      }
      else {
        // Avoid useless communications if one-column communicator and if on the last column
        if (col_comm_size > 1 && k != (mat_a.nrTiles().cols() - 1)) {
          // Receive the diagonal tile
          kk_tile = hpx::dataflow(  //
              executor_mpi,
              hpx::util::unwrapping(
                  [](auto index, auto&& tile_size, auto&& comm_wrapper) -> Tile<const T, Device::CPU> {
                    memory::MemoryView<T, Device::CPU> mem_view(
                        util::size_t::mul(tile_size.rows(), tile_size.cols()));
                    Tile<T, Device::CPU> tile(tile_size, std::move(mem_view), tile_size.rows());
                    comm::sync::broadcast::receive_from(index, comm_wrapper().colCommunicator(), tile);
                    return std::move(tile);
                  }),
              k_rank_row, mat_a.tileSize(GlobalTileIndex(k, k)), serial_comm());
        }
      }

      for (SizeType i_local = distr.nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
           i_local < localnrtile_rows; ++i_local) {
        // Update panel mat_a(i,k) with trsm (blas operation), using data mat_a.read(k,k)
        hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), Right, Lower,
                      ConjTrans, NonUnit, 1.0, kk_tile,
                      std::move(mat_a(LocalTileIndex{i_local, k_local_col})));

        // Avoid useless communications if one-row communicator grid
        if (row_comm_size > 1) {
          // Broadcast the panel row-wise
          hpx::dataflow(  //
              executor_mpi, hpx::util::unwrapping([](auto&& tile, auto&& comm_wrapper) {
                comm::sync::broadcast::send(comm_wrapper().rowCommunicator(), tile);
              }),
              mat_a.read(LocalTileIndex{i_local, k_local_col}), serial_comm());
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
          panel[i_local] = hpx::dataflow(  //
              executor_mpi,
              hpx::util::unwrapping(
                  [](auto index, auto&& tile_size, auto&& comm_wrapper) -> Tile<const T, Device::CPU> {
                    memory::MemoryView<T, Device::CPU> mem_view(
                        util::size_t::mul(tile_size.rows(), tile_size.cols()));
                    Tile<T, Device::CPU> tile(tile_size, std::move(mem_view), tile_size.rows());
                    comm::sync::broadcast::receive_from(index, comm_wrapper().rowCommunicator(), tile);
                    return std::move(tile);
                  }),
              k_rank_col, mat_a.tileSize(GlobalTileIndex(i, k)), serial_comm());
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

      hpx::shared_future<Tile<const T, Device::CPU>> col_panel;

      auto j_rank_row = distr.rankGlobalTile<Coord::Row>(j);

      if (mat_a.rankIndex().row() == j_rank_row) {
        auto i_local = distr.localTileFromGlobalTile<Coord::Row>(j);

        // Avoid useless communications if one-row communicator grid and if on the last panel
        if (col_comm_size > 1 && j != (mat_a.nrTiles().cols() - 1)) {
          // Broadcast the (trailing) panel column-wise
          hpx::dataflow(executor_mpi, hpx::util::unwrapping([](auto&& tile, auto&& comm_wrapper) {
                          comm::sync::broadcast::send(comm_wrapper().colCommunicator(), tile);
                        }),
                        panel[i_local], serial_comm());
        }

        // Check if the diagonal tile of the trailing matrix is on this node and
        // compute first tile of the column of the trailing matrix: diagonal element mat_a(j,j), reading
        // mat_a.read(j,k), using herk (blas operation)
        hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::herk<T, Device::CPU>), Lower,
                      NoTrans, -1.0, panel[i_local], 1.0, mat_a(LocalTileIndex{i_local, j_local}));

        col_panel = panel[i_local];
      }
      else {
        // Avoid useless communications if one-row communicator grid and if on the last panel
        if (col_comm_size > 1 && j != (mat_a.nrTiles().cols() - 1)) {
          // Update the (trailing) panel column-wise
          col_panel = hpx::dataflow(  //
              executor_mpi,
              hpx::util::unwrapping(
                  [](auto index, auto&& tile_size, auto&& comm_wrapper) -> Tile<const T, Device::CPU> {
                    memory::MemoryView<T, Device::CPU> mem_view(
                        util::size_t::mul(tile_size.rows(), tile_size.cols()));
                    Tile<T, Device::CPU> tile(tile_size, std::move(mem_view), tile_size.rows());
                    comm::sync::broadcast::receive_from(index, comm_wrapper().colCommunicator(), tile);
                    return std::move(tile);
                  }),
              j_rank_row, mat_a.tileSize(GlobalTileIndex(j, k)), serial_comm());
        }
      }

      for (SizeType i_local = distr.nextLocalTileFromGlobalTile<Coord::Row>(j + 1);
           i_local < localnrtile_rows; ++i_local) {
        // Update remaining trailing matrix mat_a(i,j), reading mat_a.read(i,k) and mat_a.read(j,k),
        // using gemm (blas operation)
        hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>),
                      NoTrans, ConjTrans, -1.0, panel[i_local], col_panel, 1.0,
                      std::move(mat_a(LocalTileIndex{i_local, j_local})));
      }
    }
  }
}
}
}
}
