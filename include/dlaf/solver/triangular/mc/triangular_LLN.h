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

#include <cstdlib>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
#include <hpx/util/unwrap.hpp>

#include "dlaf/blas_tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
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

// Local implementation of Left Lower NoTrans
template <class T>
void triangular_LLN(blas::Diag diag, T alpha, Matrix<const T, Device::CPU>& mat_a,
                    Matrix<T, Device::CPU>& mat_b) {
  constexpr auto Left = blas::Side::Left;
  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto NoTrans = blas::Op::NoTrans;

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

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < m; ++k) {
    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{k, j};

      // Triangular solve of k-th row Panel of B
      auto trsm_f = unwrapping([Left, Lower, NoTrans, diag, alpha](auto&& kk_tile, auto&& kj_tile) {
        tile::trsm<T, Device::CPU>(Left, Lower, NoTrans, diag, alpha, kk_tile, kj_tile);
      });
      dataflow(executor_hp, move(trsm_f), mat_a.read(LocalTileIndex{k, k}), move(mat_b(kj)));

      for (SizeType i = k + 1; i < m; ++i) {
        // Choose queue priority
        auto trailing_executor = (i == k + 1) ? executor_hp : executor_normal;
        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        auto gemm_f = unwrapping([NoTrans, beta](auto&& ik_tile, auto&& kj_tile, auto&& ij_tile) {
          tile::gemm<T, Device::CPU>(NoTrans, NoTrans, beta, ik_tile, kj_tile, 1.0, ij_tile);
        });
        dataflow(trailing_executor, move(gemm_f), mat_a.read(LocalTileIndex{i, k}), mat_b.read(kj),
                 move(mat_b(LocalTileIndex{i, j})));
      }
    }
  }
}

// Distributed implementation of Left Lower NoTrans
template <class T>
void triangular_LLN(comm::CommunicatorGrid grid, blas::Diag diag, T alpha,
                    Matrix<const T, Device::CPU>& mat_a, Matrix<T, Device::CPU>& mat_b) {
  using hpx::threads::executors::pool_executor;
  using hpx::threads::thread_priority_high;
  using hpx::threads::thread_priority_default;

  using comm::internal::mpi_pool_exists;
  using common::internal::vector;

  constexpr auto Left = blas::Side::Left;
  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto NoTrans = blas::Op::NoTrans;

  using std::move;

  using hpx::threads::executors::pool_executor;
  using hpx::threads::thread_priority_high;
  using hpx::threads::thread_priority_default;
  using hpx::util::unwrapping;
  using hpx::dataflow;

  using comm::internal::mpi_pool_exists;

  using TileType = Tile<T, Device::CPU>;
  using ConstTileType = Tile<const T, Device::CPU>;
  using MemViewType = memory::MemoryView<T, Device::CPU>;

  // Set up executor on the default queue with high priority.
  pool_executor executor_hp("default", thread_priority_high);
  // Set up executor on the default queue with default priority.
  pool_executor executor_normal("default", thread_priority_default);

  // Set up MPI executor
  auto executor_mpi = (mpi_pool_exists()) ? pool_executor("mpi", thread_priority_high) : executor_hp;

  auto col_comm_size = grid.colCommunicator().size();
  auto row_comm_size = grid.rowCommunicator().size();

  const matrix::Distribution& distr_a = mat_a.distribution();
  const matrix::Distribution& distr_b = mat_b.distribution();

  SizeType a_rows = mat_a.nrTiles().rows();

  auto a_local_rows = distr_a.localNrTiles().rows();
  auto b_local_cols = distr_b.localNrTiles().cols();

  common::Pipeline<comm::CommunicatorGrid> serial_comm(move(grid));

  for (SizeType k = 0; k < a_rows; ++k) {
    // Create a placeholder that will store the shared futures representing the panel
    vector<hpx::shared_future<ConstTileType>> panel(distr_b.localNrTiles().cols());

    auto k_rank_row = distr_a.rankGlobalTile<Coord::Row>(k);
    auto k_rank_col = distr_a.rankGlobalTile<Coord::Col>(k);

    hpx::shared_future<ConstTileType> kk_tile;

    if (mat_a.rankIndex().row() == k_rank_row) {
      auto k_local_row = distr_a.localTileFromGlobalTile<Coord::Row>(k);

      if (mat_a.rankIndex().col() == k_rank_col) {
        auto k_local_col = distr_a.localTileFromGlobalTile<Coord::Col>(k);

        auto kk = LocalTileIndex{k_local_row, k_local_col};

        // Broadcast Akk row-wise
        // Avoid useless communication if one-column communicator
        if (row_comm_size > 1) {
          auto send_f = unwrapping([](auto&& tile, auto&& comm_wrapper) {
            comm::sync::broadcast::send(comm_wrapper().rowCommunicator(), tile);
          });
          dataflow(executor_mpi, move(send_f), mat_a.read(kk), serial_comm());
        }
        kk_tile = mat_a.read(kk);
      }
      else {
        if (row_comm_size > 1) {
          auto tile_size = mat_a.tileSize(GlobalTileIndex(k, k));
          auto recv_f = unwrapping([k_rank_col, tile_size](auto&& comm_wrapper) -> ConstTileType {
            MemViewType mem_view(util::size_t::mul(tile_size.rows(), tile_size.cols()));
            TileType tile(tile_size, move(mem_view), tile_size.rows());
            comm::sync::broadcast::receive_from(k_rank_col, comm_wrapper().rowCommunicator(), tile);
            return move(tile);
          });
          kk_tile = dataflow(executor_mpi, move(recv_f), serial_comm());
        }
      }
    }

    for (SizeType j_local = 0; j_local < b_local_cols; ++j_local) {
      auto j = distr_b.globalTileFromLocalTile<Coord::Col>(j_local);

      if (mat_b.rankIndex().row() == k_rank_row) {
        auto k_local_row = distr_b.localTileFromGlobalTile<Coord::Row>(k);

        auto kj = LocalTileIndex{k_local_row, j_local};

        // Triangular solve of the k-th row Panel of B
        auto trsm_f = unwrapping([Left, Lower, NoTrans, diag, alpha](auto&& kk_tile, auto&& kj_tile) {
          tile::trsm<T, Device::CPU>(Left, Lower, NoTrans, diag, alpha, kk_tile, kj_tile);
        });
        dataflow(executor_hp, move(trsm_f), kk_tile, move(mat_b(kj)));

        // Broadcast Bkj column-wise
        // Avoid useless communication if one-column communicator and if on the last column
        if (col_comm_size > 1 && k != (mat_b.nrTiles().rows() - 1)) {
          auto send_f = unwrapping([](auto&& tile, auto&& comm_wrapper) {
            comm::sync::broadcast::send(comm_wrapper().colCommunicator(), tile);
          });
          dataflow(executor_mpi, move(send_f), mat_b.read(kj), serial_comm());
        }
        panel[j_local] = mat_b.read(kj);
      }
      else {
        if (col_comm_size > 1 && k != (mat_b.nrTiles().rows() - 1)) {
          auto tile_size = mat_b.tileSize(GlobalTileIndex(k, j));
          auto recv_f = unwrapping([k_rank_row, tile_size](auto&& comm_wrapper) -> ConstTileType {
            MemViewType mem_view(util::size_t::mul(tile_size.rows(), tile_size.cols()));
            TileType tile(tile_size, move(mem_view), tile_size.rows());
            comm::sync::broadcast::receive_from(k_rank_row, comm_wrapper().colCommunicator(), tile);
            return move(tile);
          });
          panel[j_local] = dataflow(executor_mpi, move(recv_f), serial_comm());
        }
      }
    }

    for (SizeType i_local = distr_a.nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
         i_local < a_local_rows; ++i_local) {
      auto i = distr_a.globalTileFromLocalTile<Coord::Row>(i_local);

      // Choose queue priority
      auto trailing_executor = (i == k + 1) ? executor_hp : executor_normal;

      hpx::shared_future<Tile<const T, Device::CPU>> ik_tile;

      // Broadcast Aik row-wise
      if (mat_a.rankIndex().col() == k_rank_col) {
        auto k_local_col = distr_a.localTileFromGlobalTile<Coord::Col>(k);

        auto ik = LocalTileIndex{i_local, k_local_col};

        if (row_comm_size > 1) {
          auto send_f = unwrapping([](auto&& tile, auto&& comm_wrapper) {
            comm::sync::broadcast::send(comm_wrapper().rowCommunicator(), tile);
          });
          dataflow(executor_mpi, move(send_f), mat_a.read(ik), serial_comm());
        }
        ik_tile = mat_a.read(ik);
      }
      else {
        if (row_comm_size > 1) {
          auto tile_size = mat_a.tileSize(GlobalTileIndex(i, k));
          auto recv_f = unwrapping([k_rank_col, tile_size](auto&& comm_wrapper) -> ConstTileType {
            memory::MemoryView<T, Device::CPU> mem_view(
                util::size_t::mul(tile_size.rows(), tile_size.cols()));
            Tile<T, Device::CPU> tile(tile_size, move(mem_view), tile_size.rows());
            comm::sync::broadcast::receive_from(k_rank_col, comm_wrapper().rowCommunicator(), tile);
            return move(tile);
          });
          ik_tile = dataflow(executor_mpi, move(recv_f), serial_comm());
        }
      }

      for (SizeType j_local = 0; j_local < b_local_cols; ++j_local) {
        // Update trailing matrix
        auto beta = static_cast<T>(-1.0) / alpha;
        auto gemm_f = unwrapping([NoTrans, beta](auto&& ik_tile, auto&& kj_tile, auto&& ij_tile) {
          tile::gemm<T, Device::CPU>(NoTrans, NoTrans, beta, ik_tile, kj_tile, 1.0, ij_tile);
        });
        dataflow(trailing_executor, move(gemm_f), ik_tile, panel[j_local],
                 move(mat_b(LocalTileIndex{i_local, j_local})));
      }
    }
  }
}

}
}
}
