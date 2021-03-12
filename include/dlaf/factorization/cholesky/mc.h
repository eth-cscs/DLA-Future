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

#include <unordered_map>

#include "dlaf/blas_tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/executors.h"
#include "dlaf/factorization/cholesky/api.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix/distribution.h"

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
  hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), blas::Side::Right,
                blas::Uplo::Lower, blas::Op::ConjTrans, blas::Diag::NonUnit, 1.0, std::move(kk_tile),
                std::move(matrix_tile));
}

template <class T>
void herk_trailing_diag_tile(hpx::execution::parallel_executor trailing_matrix_executor,
                             hpx::shared_future<matrix::Tile<const T, Device::CPU>> panel_tile,
                             hpx::future<matrix::Tile<T, Device::CPU>> matrix_tile) {
  hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::herk<T, Device::CPU>),
                blas::Uplo::Lower, blas::Op::NoTrans, -1.0, panel_tile, 1.0, std::move(matrix_tile));
}

template <class T>
void gemm_trailing_matrix_tile(hpx::execution::parallel_executor trailing_matrix_executor,
                               hpx::shared_future<matrix::Tile<const T, Device::CPU>> panel_tile,
                               hpx::shared_future<matrix::Tile<const T, Device::CPU>> col_panel,
                               hpx::future<matrix::Tile<T, Device::CPU>> matrix_tile) {
  hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>),
                blas::Op::NoTrans, blas::Op::ConjTrans, -1.0, std::move(panel_tile),
                std::move(col_panel), 1.0, std::move(matrix_tile));
}

template <class T>
struct Cholesky<Backend::MC, Device::CPU, T> {
  static void call_L(Matrix<T, Device::CPU>& mat_a);
  static void call_L(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_a);
};

template <class T>
void Cholesky<Backend::MC, Device::CPU, T>::call_L(Matrix<T, Device::CPU>& mat_a) {
  auto executor_hp = dlaf::getHpExecutor<Backend::MC>();
  auto executor_np = dlaf::getNpExecutor<Backend::MC>();

  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().cols();

  for (SizeType k = 0; k < nrtile; ++k) {
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
void Cholesky<Backend::MC, Device::CPU, T>::call_L(comm::CommunicatorGrid grid,
                                                   Matrix<T, Device::CPU>& mat_a) {
  using ConstTileType = typename Matrix<T, Device::CPU>::ConstTileType;

  auto executor_hp = dlaf::getHpExecutor<Backend::MC>();
  auto executor_np = dlaf::getNpExecutor<Backend::MC>();
  auto executor_mpi = dlaf::getMPIExecutor<Backend::MC>();

  common::Pipeline<comm::CommunicatorGrid> mpi_task_chain(std::move(grid));

  const comm::Index2D this_rank = grid.rank();

  matrix::Distribution const& distr = mat_a.distribution();
  SizeType nrtile = mat_a.nrTiles().cols();

  for (SizeType k = 0; k < nrtile; ++k) {
    // Create a placeholder that will store the shared futures representing the panel
    std::unordered_map<SizeType, hpx::shared_future<ConstTileType>> panel;

    GlobalTileIndex kk_idx(k, k);
    comm::Index2D kk_rank = distr.rankGlobalTile(kk_idx);

    // Broadcast the diagonal tile along the `k`-th column
    if (this_rank == kk_rank) {
      potrf_diag_tile(executor_hp, mat_a(kk_idx));
      panel[k] = mat_a.read(kk_idx);
      if (k != nrtile - 1)
        dataflow(executor_mpi, comm::sendTile_o, mpi_task_chain(), Coord::Col, panel[k]);
    }
    else if (this_rank.col() == kk_rank.col()) {
      if (k != nrtile - 1)
        panel[k] = dataflow(executor_mpi, comm::recvAllocTile<T>, mpi_task_chain(), Coord::Col,
                            mat_a.tileSize(kk_idx), kk_rank.row());
    }

    // Iterate over the k-th column
    for (SizeType i = k + 1; i < nrtile; ++i) {
      GlobalTileIndex ik_idx(i, k);
      comm::Index2D ik_rank = mat_a.rankGlobalTile(ik_idx);

      if (this_rank == ik_rank) {
        trsm_panel_tile(executor_hp, panel[k], mat_a(ik_idx));
        panel[i] = mat_a.read(ik_idx);
        dataflow(executor_mpi, comm::sendTile_o, mpi_task_chain(), Coord::Row, panel[i]);
      }
      else if (this_rank.row() == ik_rank.row()) {
        panel[i] = dataflow(executor_mpi, comm::recvAllocTile<T>, mpi_task_chain(), Coord::Row,
                            mat_a.tileSize(ik_idx), ik_rank.col());
      }
    }

    // Iterate over the trailing matrix
    for (SizeType j = k + 1; j < nrtile; ++j) {
      GlobalTileIndex jj_idx(j, j);
      comm::Index2D jj_rank = mat_a.rankGlobalTile(jj_idx);

      if (this_rank.col() != jj_rank.col())
        continue;

      // Broadcast the jk-tile along the j-th column and update the jj-tile
      if (this_rank.row() == jj_rank.row()) {
        auto& trailing_matrix_executor = (j == k + 1) ? executor_hp : executor_np;
        herk_trailing_diag_tile(trailing_matrix_executor, panel[j], mat_a(jj_idx));
        if (j != nrtile - 1)
          dataflow(executor_mpi, comm::sendTile_o, mpi_task_chain(), Coord::Col, panel[j]);
      }
      else {
        GlobalTileIndex jk_idx(j, k);
        if (j != nrtile - 1)
          panel[j] = dataflow(executor_mpi, comm::recvAllocTile<T>, mpi_task_chain(), Coord::Col,
                              mat_a.tileSize(jk_idx), jj_rank.row());
      }

      for (SizeType i = j + 1; i < nrtile; ++i) {
        // Update the ij-tile using the ik-tile and jk-tile
        if (this_rank.row() == distr.rankGlobalTile<Coord::Row>(i)) {
          GlobalTileIndex ij_idx(i, j);
          gemm_trailing_matrix_tile(executor_np, panel[i], panel[j], mat_a(ij_idx));
        }
      }
    }
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
