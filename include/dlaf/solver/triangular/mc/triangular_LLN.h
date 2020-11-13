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
#include <hpx/include/util.hpp>
#include <hpx/local/future.hpp>

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

  using hpx::threads::executors::pool_executor;
  using hpx::threads::thread_priority_high;
  using hpx::threads::thread_priority_default;

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
      hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), Left, Lower, NoTrans,
                    diag, alpha, mat_a.read(LocalTileIndex{k, k}), std::move(mat_b(kj)));

      for (SizeType i = k + 1; i < m; ++i) {
        // Choose queue priority
        auto trailing_executor = (i == k + 1) ? executor_hp : executor_normal;
        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
                      NoTrans, beta, mat_a.read(LocalTileIndex{i, k}), mat_b.read(kj), 1.0,
                      std::move(mat_b(LocalTileIndex{i, j})));
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
  using ConstTileType = typename Matrix<T, Device::CPU>::ConstTileType;

  // Set up executor on the default queue with high priority.
  pool_executor executor_hp("default", thread_priority_high);
  // Set up executor on the default queue with default priority.
  pool_executor executor_normal("default", thread_priority_default);
  // Set up MPI executor
  auto executor_mpi = (mpi_pool_exists()) ? pool_executor("mpi", thread_priority_high) : executor_hp;

  const matrix::Distribution& distr_a = mat_a.distribution();
  const matrix::Distribution& distr_b = mat_b.distribution();

  SizeType a_rows = mat_a.nrTiles().rows();

  auto a_local_rows = distr_a.localNrTiles().rows();
  auto b_local_cols = distr_b.localNrTiles().cols();

  common::Pipeline<comm::CommunicatorGrid> serial_comm(std::move(grid));

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
        comm::send_tile(executor_mpi, serial_comm, Coord::Row, mat_a.read(kk));
        kk_tile = mat_a.read(kk);
      }
      else {
        kk_tile = comm::recv_tile<T>(executor_mpi, serial_comm, Coord::Row,
                                     mat_a.tileSize(GlobalTileIndex(k, k)), k_rank_col);
      }
    }

    for (SizeType j_local = 0; j_local < b_local_cols; ++j_local) {
      auto j = distr_b.globalTileFromLocalTile<Coord::Col>(j_local);

      if (mat_b.rankIndex().row() == k_rank_row) {
        auto k_local_row = distr_b.localTileFromGlobalTile<Coord::Row>(k);

        auto kj = LocalTileIndex{k_local_row, j_local};

        // Triangular solve of the k-th row Panel of B
        hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), Left, Lower,
                      NoTrans, diag, alpha, kk_tile, std::move(mat_b(kj)));

        // Broadcast Bkj column-wise
        // Avoid useless communication if one-column communicator and if on the last column
        panel[j_local] = mat_b.read(kj);
        if (k != (mat_b.nrTiles().rows() - 1)) {
          comm::send_tile(executor_mpi, serial_comm, Coord::Col, panel[j_local]);
        }
      }
      else {
        if (k != (mat_b.nrTiles().rows() - 1)) {
          panel[j_local] = comm::recv_tile<T>(executor_mpi, serial_comm, Coord::Col,
                                              mat_b.tileSize(GlobalTileIndex(k, j)), k_rank_row);
        }
      }
    }

    for (SizeType i_local = distr_a.nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
         i_local < a_local_rows; ++i_local) {
      auto i = distr_a.globalTileFromLocalTile<Coord::Row>(i_local);

      // Choose queue priority
      auto trailing_executor = (i == k + 1) ? executor_hp : executor_normal;

      hpx::shared_future<ConstTileType> ik_tile;

      // Broadcast Aik row-wise
      if (mat_a.rankIndex().col() == k_rank_col) {
        auto k_local_col = distr_a.localTileFromGlobalTile<Coord::Col>(k);
        auto ik = LocalTileIndex{i_local, k_local_col};
        ik_tile = mat_a.read(ik);
        comm::send_tile(executor_mpi, serial_comm, Coord::Row, mat_a.read(ik));
      }
      else {
        ik_tile = comm::recv_tile<T>(executor_mpi, serial_comm, Coord::Row,
                                     mat_a.tileSize(GlobalTileIndex(i, k)), k_rank_col);
      }

      for (SizeType j_local = 0; j_local < b_local_cols; ++j_local) {
        // Update trailing matrix
        auto beta = static_cast<T>(-1.0) / alpha;
        hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
                      NoTrans, beta, ik_tile, panel[j_local], 1.0,
                      std::move(mat_b(LocalTileIndex{i_local, j_local})));
      }
    }
  }
}

}
}
}
