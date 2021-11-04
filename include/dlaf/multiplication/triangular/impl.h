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

#include <hpx/local/future.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/communication/broadcast_panel.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/executors.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/multiplication/triangular/api.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace multiplication {
namespace internal {

namespace triangular_lln {
template <class Executor, class T, Device D>
void trmm_B_panel_tile(Executor&& ex, blas::Diag diag, T alpha,
                       hpx::shared_future<matrix::Tile<const T, D>> in_tile,
                       hpx::future<matrix::Tile<T, D>> out_tile) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(tile::trmm_o), blas::Side::Left,
                blas::Uplo::Lower, blas::Op::NoTrans, diag, alpha, std::move(in_tile),
                std::move(out_tile));
}

template <class Executor, class T, Device D>
void gemm_trailing_matrix_tile(Executor&& ex, T beta,
                               hpx::shared_future<matrix::Tile<const T, D>> a_tile,
                               hpx::shared_future<matrix::Tile<const T, D>> b_tile,
                               hpx::future<matrix::Tile<T, D>> c_tile) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::NoTrans,
                blas::Op::NoTrans, beta, std::move(a_tile), std::move(b_tile), T(1.0),
                std::move(c_tile));
}
}

namespace triangular_llt {
template <class Executor, class T, Device D>
void trmm_B_panel_tile(Executor&& ex, blas::Op op, blas::Diag diag, T alpha,
                       hpx::shared_future<matrix::Tile<const T, D>> in_tile,
                       hpx::future<matrix::Tile<T, D>> out_tile) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(tile::trmm_o), blas::Side::Left,
                blas::Uplo::Lower, op, diag, alpha, std::move(in_tile), std::move(out_tile));
}

template <class Executor, class T, Device D>
void gemm_trailing_matrix_tile(Executor&& ex, blas::Op op, T alpha,
                               hpx::shared_future<matrix::Tile<const T, D>> a_tile,
                               hpx::shared_future<matrix::Tile<const T, D>> b_tile,
                               hpx::future<matrix::Tile<T, D>> c_tile) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(tile::gemm_o), op,
                blas::Op::NoTrans, alpha, std::move(a_tile), std::move(b_tile), T(1.0),
                std::move(c_tile));
}
}

namespace triangular_lun {
template <class Executor, class T, Device D>
void trmm_B_panel_tile(Executor&& ex, blas::Diag diag, T alpha,
                       hpx::shared_future<matrix::Tile<const T, D>> in_tile,
                       hpx::future<matrix::Tile<T, D>> out_tile) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(tile::trmm_o), blas::Side::Left,
                blas::Uplo::Upper, blas::Op::NoTrans, diag, alpha, std::move(in_tile),
                std::move(out_tile));
}

template <class Executor, class T, Device D>
void gemm_trailing_matrix_tile(Executor&& ex, T alpha,
                               hpx::shared_future<matrix::Tile<const T, D>> a_tile,
                               hpx::shared_future<matrix::Tile<const T, D>> b_tile,
                               hpx::future<matrix::Tile<T, D>> c_tile) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::NoTrans,
                blas::Op::NoTrans, alpha, std::move(a_tile), std::move(b_tile), T(1.0),
                std::move(c_tile));
}
}

namespace triangular_lut {
template <class Executor, class T, Device D>
void trmm_B_panel_tile(Executor&& ex, blas::Op op, blas::Diag diag, T alpha,
                       hpx::shared_future<matrix::Tile<const T, D>> in_tile,
                       hpx::future<matrix::Tile<T, D>> out_tile) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(tile::trmm_o), blas::Side::Left,
                blas::Uplo::Upper, op, diag, alpha, std::move(in_tile), std::move(out_tile));
}

template <class Executor, class T, Device D>
void gemm_trailing_matrix_tile(Executor&& ex, blas::Op op, T alpha,
                               hpx::shared_future<matrix::Tile<const T, D>> a_tile,
                               hpx::shared_future<matrix::Tile<const T, D>> b_tile,
                               hpx::future<matrix::Tile<T, D>> c_tile) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(tile::gemm_o), op,
                blas::Op::NoTrans, alpha, std::move(a_tile), std::move(b_tile), T(1.0),
                std::move(c_tile));
}
}

namespace triangular_rln {
template <class Executor, class T, Device D>
void trmm_B_panel_tile(Executor&& ex, blas::Diag diag, T alpha,
                       hpx::shared_future<matrix::Tile<const T, D>> in_tile,
                       hpx::future<matrix::Tile<T, D>> out_tile) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(tile::trmm_o), blas::Side::Right,
                blas::Uplo::Lower, blas::Op::NoTrans, diag, alpha, std::move(in_tile),
                std::move(out_tile));
}

template <class Executor, class T, Device D>
void gemm_trailing_matrix_tile(Executor&& ex, T alpha,
                               hpx::shared_future<matrix::Tile<const T, D>> a_tile,
                               hpx::shared_future<matrix::Tile<const T, D>> b_tile,
                               hpx::future<matrix::Tile<T, D>> c_tile) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::NoTrans,
                blas::Op::NoTrans, alpha, std::move(a_tile), std::move(b_tile), T(1.0),
                std::move(c_tile));
}
}

namespace triangular_rlt {
template <class Executor, class T, Device D>
void trmm_B_panel_tile(Executor&& ex, blas::Op op, blas::Diag diag, T alpha,
                       hpx::shared_future<matrix::Tile<const T, D>> in_tile,
                       hpx::future<matrix::Tile<T, D>> out_tile) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(tile::trmm_o), blas::Side::Right,
                blas::Uplo::Lower, op, diag, alpha, std::move(in_tile), std::move(out_tile));
}

template <class Executor, class T, Device D>
void gemm_trailing_matrix_tile(Executor&& ex, blas::Op op, T alpha,
                               hpx::shared_future<matrix::Tile<const T, D>> a_tile,
                               hpx::shared_future<matrix::Tile<const T, D>> b_tile,
                               hpx::future<matrix::Tile<T, D>> c_tile) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::NoTrans,
                op, alpha, std::move(a_tile), std::move(b_tile), T(1.0), std::move(c_tile));
}
}

namespace triangular_run {
template <class Executor, class T, Device D>
void trmm_B_panel_tile(Executor&& ex, blas::Diag diag, T alpha,
                       hpx::shared_future<matrix::Tile<const T, D>> in_tile,
                       hpx::future<matrix::Tile<T, D>> out_tile) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(tile::trmm_o), blas::Side::Right,
                blas::Uplo::Upper, blas::Op::NoTrans, diag, alpha, std::move(in_tile),
                std::move(out_tile));
}

template <class Executor, class T, Device D>
void gemm_trailing_matrix_tile(Executor&& ex, T alpha,
                               hpx::shared_future<matrix::Tile<const T, D>> a_tile,
                               hpx::shared_future<matrix::Tile<const T, D>> b_tile,
                               hpx::future<matrix::Tile<T, D>> c_tile) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::NoTrans,
                blas::Op::NoTrans, alpha, std::move(a_tile), std::move(b_tile), T(1.0),
                std::move(c_tile));
}
}

namespace triangular_rut {
template <class Executor, class T, Device D>
void trmm_B_panel_tile(Executor&& ex, blas::Op op, blas::Diag diag, T alpha,
                       hpx::shared_future<matrix::Tile<const T, D>> in_tile,
                       hpx::future<matrix::Tile<T, D>> out_tile) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(tile::trmm_o), blas::Side::Right,
                blas::Uplo::Upper, op, diag, alpha, std::move(in_tile), std::move(out_tile));
}

template <class Executor, class T, Device D>
void gemm_trailing_matrix_tile(Executor&& ex, blas::Op op, T alpha,
                               hpx::shared_future<matrix::Tile<const T, D>> a_tile,
                               hpx::shared_future<matrix::Tile<const T, D>> b_tile,
                               hpx::future<matrix::Tile<T, D>> c_tile) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::NoTrans,
                op, alpha, std::move(a_tile), std::move(b_tile), T(1.0), std::move(c_tile));
}
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LLN(blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                                              Matrix<T, device>& mat_b) {
  using namespace triangular_lln;
  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();
  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = m - 1; k >= 0; --k) {
    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{k, j};

      for (SizeType i = k + 1; i < m; ++i) {
        auto& trailing_executor = (i == k + 1) ? executor_hp : executor_np;
        gemm_trailing_matrix_tile(trailing_executor, alpha, mat_a.read(LocalTileIndex{i, k}),
                                  mat_b.read(kj), mat_b(LocalTileIndex{i, j}));
      }

      trmm_B_panel_tile(executor_hp, diag, alpha, mat_a.read(LocalTileIndex{k, k}), mat_b(kj));
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LLT(blas::Op op, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_llt;
  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < m; ++k) {
    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{k, j};

      for (SizeType i = k - 1; i >= 0; --i) {
        auto& trailing_executor = (i == k - 1) ? executor_hp : executor_np;
        gemm_trailing_matrix_tile(trailing_executor, op, alpha, mat_a.read(LocalTileIndex{k, i}),
                                  mat_b.read(kj), mat_b(LocalTileIndex{i, j}));
      }

      trmm_B_panel_tile(executor_hp, op, diag, alpha, mat_a.read(LocalTileIndex{k, k}), mat_b(kj));
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LUN(blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                                              Matrix<T, device>& mat_b) {
  using namespace triangular_lun;
  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < m; ++k) {
    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{k, j};

      for (SizeType i = k - 1; i >= 0; --i) {
        auto& trailing_executor = (i == k - 1) ? executor_hp : executor_np;
        gemm_trailing_matrix_tile(trailing_executor, alpha, mat_a.read(LocalTileIndex{i, k}),
                                  mat_b.read(kj), mat_b(LocalTileIndex{i, j}));
      }

      trmm_B_panel_tile(executor_hp, diag, alpha, mat_a.read(LocalTileIndex{k, k}), mat_b(kj));
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LUT(blas::Op op, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_lut;
  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = m - 1; k >= 0; --k) {
    for (SizeType j = n - 1; j >= 0; --j) {
      auto kj = LocalTileIndex{k, j};

      for (SizeType i = k + 1; i < m; ++i) {
        auto& trailing_executor = (i == k + 1) ? executor_hp : executor_np;
        gemm_trailing_matrix_tile(trailing_executor, op, alpha, mat_a.read(LocalTileIndex{k, i}),
                                  mat_b.read(kj), mat_b(LocalTileIndex{i, j}));
      }

      trmm_B_panel_tile(executor_hp, op, diag, alpha, mat_a.read(LocalTileIndex{k, k}), mat_b(kj));
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RLN(blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                                              Matrix<T, device>& mat_b) {
  using namespace triangular_rln;
  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < n; ++k) {
    for (SizeType i = 0; i < m; ++i) {
      auto ik = LocalTileIndex{i, k};

      for (SizeType j = k - 1; j >= 0; --j) {
        auto& trailing_executor = (j == k - 1) ? executor_hp : executor_np;
        gemm_trailing_matrix_tile(trailing_executor, alpha, mat_b.read(ik),
                                  mat_a.read(LocalTileIndex{k, j}), mat_b(LocalTileIndex{i, j}));
      }

      trmm_B_panel_tile(executor_hp, diag, alpha, mat_a.read(LocalTileIndex{k, k}), mat_b(ik));
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RLT(blas::Op op, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_rlt;
  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = n - 1; k >= 0; --k) {
    for (SizeType i = m - 1; i >= 0; --i) {
      auto ik = LocalTileIndex{i, k};

      for (SizeType j = k + 1; j < n; ++j) {
        auto& trailing_executor = (j == k + 1) ? executor_hp : executor_np;
        gemm_trailing_matrix_tile(trailing_executor, op, alpha, mat_b.read(ik),
                                  mat_a.read(LocalTileIndex{j, k}), mat_b(LocalTileIndex{i, j}));
      }

      trmm_B_panel_tile(executor_hp, op, diag, alpha, mat_a.read(LocalTileIndex{k, k}), mat_b(ik));
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RUN(blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                                              Matrix<T, device>& mat_b) {
  using namespace triangular_run;
  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = n - 1; k >= 0; --k) {
    for (SizeType i = m - 1; i >= 0; --i) {
      auto ik = LocalTileIndex{i, k};

      for (SizeType j = k + 1; j < n; ++j) {
        auto& trailing_executor = (j == k + 1) ? executor_hp : executor_np;
        gemm_trailing_matrix_tile(trailing_executor, alpha, mat_b.read(ik),
                                  mat_a.read(LocalTileIndex{k, j}), mat_b(LocalTileIndex{i, j}));
      }

      trmm_B_panel_tile(executor_hp, diag, alpha, mat_a.read(LocalTileIndex{k, k}), mat_b(ik));
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RUT(blas::Op op, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_rut;
  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < n; ++k) {
    for (SizeType i = 0; i < m; ++i) {
      auto ik = LocalTileIndex{i, k};

      for (SizeType j = k - 1; j >= 0; --j) {
        auto& trailing_executor = (j == k - 1) ? executor_hp : executor_np;
        gemm_trailing_matrix_tile(trailing_executor, op, alpha, mat_b.read(ik),
                                  mat_a.read(LocalTileIndex{j, k}), mat_b(LocalTileIndex{i, j}));
      }

      trmm_B_panel_tile(executor_hp, op, diag, alpha, mat_a.read(LocalTileIndex{k, k}), mat_b(ik));
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LLN(comm::CommunicatorGrid grid, blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                                              Matrix<T, device>& mat_b) {
  using namespace triangular_lln;

  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();
  auto executor_mpi = dlaf::getMPIExecutor<backend>();

  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr_a = mat_a.distribution();
  const matrix::Distribution& distr_b = mat_b.distribution();

  if (mat_b.size().isEmpty())
    return;

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> a_panels(n_workspaces, distr_a);
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> b_panels(n_workspaces, distr_b);

  
  for (SizeType k = mat_a.nrTiles().rows() - 1; k >= 0; --k) {
    const GlobalTileIndex kk(k, k);
    auto kk_rank = distr_a.rankGlobalTile(kk);

    const LocalTileIndex kk_offset{
        distr_a.nextLocalTileFromGlobalTile<Coord::Row>(k),
        distr_a.nextLocalTileFromGlobalTile<Coord::Col>(k),
    };

    const LocalTileIndex bt_offset{distr_a.nextLocalTileFromGlobalTile<Coord::Row>(k + 1), 0};

    auto& a_panel = a_panels.nextResource();
    auto& b_panel = b_panels.nextResource();
    a_panel.setRangeStart(kk);
    if (k == mat_a.nrTiles().cols() - 1) {
      a_panel.setWidth(mat_a.tileSize(kk).rows());
      b_panel.setHeight(mat_a.tileSize(kk).cols());
    }
    
    if (kk_rank.col() == this_rank.col()) {
      for (SizeType i_local = kk_offset.row(); i_local < distr_a.localNrTiles().rows(); ++i_local) {
        const LocalTileIndex ik_panel(Coord::Row, i_local);
        const LocalTileIndex ik(i_local, kk_offset.col());

        a_panel.setTile(ik_panel, mat_a.read(ik));
      }
    }
    broadcast(executor_mpi, kk_rank.col(), a_panel, mpi_row_task_chain);

    for (SizeType j_local = 0; j_local < distr_b.localNrTiles().cols(); ++j_local) {
      if (kk_rank.row() == this_rank.row()) {
        auto k_local_row = distr_b.localTileFromGlobalTile<Coord::Row>(k);
	const LocalTileIndex kk_panel(Coord::Row, k_local_row);
        const LocalTileIndex kj(k_local_row, j_local);
	const LocalTileIndex kj_panel(Coord::Col, j_local);

        b_panel.setTile(kj_panel, mat_b.read(kj));
	trmm_B_panel_tile(executor_hp, diag, alpha, a_panel.read(kk_panel), mat_b(kj));
      }
    }

    broadcast(executor_mpi, kk_rank.row(), b_panel, mpi_col_task_chain);

    for (SizeType i_local = bt_offset.row(); i_local < distr_a.localNrTiles().rows(); ++i_local) {
      // Choose queue priority
      auto i = distr_a.globalTileFromLocalTile<Coord::Row>(i_local);
      auto& trailing_executor = (i == k + 1) ? executor_hp : executor_np;
      
      const LocalTileIndex ik_panel(Coord::Row, i_local);
      // Update trailing matrix
      for (SizeType j_local = 0; j_local < distr_b.localNrTiles().cols(); ++j_local) {

        const LocalTileIndex kj_panel(Coord::Col, j_local);
        const LocalTileIndex ij(i_local, j_local);
        gemm_trailing_matrix_tile(trailing_executor, alpha, a_panel.read(ik_panel),
                                  b_panel.read(kj_panel), mat_b(ij));
      }
    }
    
    a_panel.reset();
    b_panel.reset();
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LUN(comm::CommunicatorGrid grid, blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                                              Matrix<T, device>& mat_b) {
  using namespace triangular_lun;

  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();
  auto executor_mpi = dlaf::getMPIExecutor<backend>();

  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr_a = mat_a.distribution();
  const matrix::Distribution& distr_b = mat_b.distribution();

  if (mat_b.size().isEmpty())
    return;

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> a_panels(n_workspaces, distr_a);
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> b_panels(n_workspaces, distr_b);

  
  for (SizeType k = 0; k < mat_a.nrTiles().rows(); ++k) {
    const GlobalTileIndex kk(k, k);
    auto kk_rank = distr_a.rankGlobalTile(kk);

    const LocalTileIndex kk_offset{
        distr_a.nextLocalTileFromGlobalTile<Coord::Row>(k + 1),
        distr_a.nextLocalTileFromGlobalTile<Coord::Col>(k),
    };

    const LocalTileIndex bt_offset{distr_a.nextLocalTileFromGlobalTile<Coord::Row>(k), 0};

    auto& a_panel = a_panels.nextResource();
    auto& b_panel = b_panels.nextResource();
    if (k == mat_a.nrTiles().cols() - 1) {
      a_panel.setWidth(mat_a.tileSize(kk).rows());
      b_panel.setHeight(mat_a.tileSize(kk).cols());
    }
    
    if (kk_rank.col() == this_rank.col()) {
      for (SizeType i_local = kk_offset.row() - 1; i_local >= 0; --i_local) {
        const LocalTileIndex ik_panel(Coord::Row, i_local);
        const LocalTileIndex ik(i_local, kk_offset.col());

        a_panel.setTile(ik_panel, mat_a.read(ik));
      }
    }
    broadcast(executor_mpi, kk_rank.col(), a_panel, mpi_row_task_chain);

    for (SizeType j_local = 0; j_local < distr_b.localNrTiles().cols(); ++j_local) {
      if (kk_rank.row() == this_rank.row()) {
        auto k_local_row = distr_b.localTileFromGlobalTile<Coord::Row>(k);
	const LocalTileIndex kk_panel(Coord::Row, k_local_row);
        const LocalTileIndex kj(k_local_row, j_local);
	const LocalTileIndex kj_panel(Coord::Col, j_local);

        b_panel.setTile(kj_panel, mat_b.read(kj));
	trmm_B_panel_tile(executor_hp, diag, alpha, a_panel.read(kk_panel), mat_b(kj));
      }
    }
    broadcast(executor_mpi, kk_rank.row(), b_panel, mpi_col_task_chain);

    for (SizeType i_local = bt_offset.row() - 1; i_local >= 0; --i_local) {
      // Choose queue priority
      auto i = distr_a.globalTileFromLocalTile<Coord::Row>(i_local);
      auto& trailing_executor = (i == k - 1) ? executor_hp : executor_np;

      const LocalTileIndex ik_panel(Coord::Row, i_local);
      // Update trailing matrix
      for (SizeType j_local = 0; j_local < distr_b.localNrTiles().cols(); ++j_local) {
        const LocalTileIndex kj_panel(Coord::Col, j_local);
        const LocalTileIndex ij(i_local, j_local);
        gemm_trailing_matrix_tile(trailing_executor, alpha, a_panel.read(ik_panel),
                                  b_panel.read(kj_panel), mat_b(ij));
      }
    }
    
    a_panel.reset();
    b_panel.reset();
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RLN(comm::CommunicatorGrid grid, blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                                              Matrix<T, device>& mat_b) {
  using namespace triangular_rln;

  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();
  auto executor_mpi = dlaf::getMPIExecutor<backend>();
  
  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr_a = mat_a.distribution();
  const matrix::Distribution& distr_b = mat_b.distribution();

  if (mat_b.size().isEmpty())
    return;

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> a_panels(n_workspaces, distr_a);
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> b_panels(n_workspaces, distr_b);

  for (SizeType k = 0; k < mat_a.nrTiles().cols(); ++k) {
    const GlobalTileIndex kk(k, k);
    auto kk_rank = distr_a.rankGlobalTile(kk);

    const LocalTileIndex kk_offset{
        distr_a.nextLocalTileFromGlobalTile<Coord::Row>(k),
        distr_a.nextLocalTileFromGlobalTile<Coord::Col>(k + 1),
    };

    const LocalTileIndex bt_offset{0, distr_a.nextLocalTileFromGlobalTile<Coord::Col>(k)};

    auto& a_panel = a_panels.nextResource();
    auto& b_panel = b_panels.nextResource();
    if (k == mat_a.nrTiles().cols() - 1) {
      a_panel.setHeight(mat_a.tileSize(kk).cols());
      b_panel.setWidth(mat_a.tileSize(kk).rows());
    }
    
    if (kk_rank.row() == this_rank.row()) {
      for (SizeType j_local = kk_offset.col() - 1; j_local >= 0; --j_local) {
        const LocalTileIndex kj_panel(Coord::Col, j_local);
        const LocalTileIndex kj(kk_offset.row(), j_local);

        a_panel.setTile(kj_panel, mat_a.read(kj));
      }
    }
    broadcast(executor_mpi, kk_rank.row(), a_panel, mpi_col_task_chain);

    for (SizeType i_local = 0; i_local < distr_b.localNrTiles().rows(); ++i_local) {
      if (kk_rank.col() == this_rank.col()) {
        auto k_local_col = distr_b.localTileFromGlobalTile<Coord::Col>(k);
	const LocalTileIndex kk_panel(Coord::Col, k_local_col);
        const LocalTileIndex ik(i_local, k_local_col);
	const LocalTileIndex ik_panel(Coord::Row, i_local);

        b_panel.setTile(ik_panel, mat_b.read(ik));
		trmm_B_panel_tile(executor_hp, diag, alpha, a_panel.read(kk_panel), mat_b(ik));
      }
    }

    broadcast(executor_mpi, kk_rank.col(), b_panel, mpi_row_task_chain);

    for (SizeType j_local = bt_offset.col() - 1; j_local >= 0; --j_local) {
      // Choose queue priority
      auto j = distr_a.globalTileFromLocalTile<Coord::Col>(j_local);
      auto& trailing_executor = (j == k - 1) ? executor_hp : executor_np;
      
      const LocalTileIndex kj_panel(Coord::Col, j_local);
      // Update trailing matrix
      for (SizeType i_local = 0; i_local < distr_b.localNrTiles().rows(); ++i_local) {

        const LocalTileIndex ik_panel(Coord::Row, i_local);
        const LocalTileIndex ij(i_local, j_local);
        gemm_trailing_matrix_tile(trailing_executor, alpha, b_panel.read(ik_panel),
                                  a_panel.read(kj_panel), mat_b(ij));
      }
    }

    a_panel.reset();
    b_panel.reset();
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RUN(comm::CommunicatorGrid grid, blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                                              Matrix<T, device>& mat_b) {
  using namespace triangular_run;

  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();
  auto executor_mpi = dlaf::getMPIExecutor<backend>();

  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr_a = mat_a.distribution();
  const matrix::Distribution& distr_b = mat_b.distribution();

  if (mat_b.size().isEmpty())
    return;

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> a_panels(n_workspaces, distr_a);
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> b_panels(n_workspaces, distr_b);

  
  for (SizeType k = mat_a.nrTiles().cols() - 1; k >= 0; --k) {
    const GlobalTileIndex kk(k, k);
    auto kk_rank = distr_a.rankGlobalTile(kk);

    const LocalTileIndex kk_offset{
        distr_a.nextLocalTileFromGlobalTile<Coord::Row>(k),
        distr_a.nextLocalTileFromGlobalTile<Coord::Col>(k),
    };

    const LocalTileIndex bt_offset{0, distr_a.nextLocalTileFromGlobalTile<Coord::Col>(k + 1)};

    auto& a_panel = a_panels.nextResource();
    auto& b_panel = b_panels.nextResource();
    a_panel.setRangeStart(kk);
    if (k == mat_a.nrTiles().cols() - 1) {
      a_panel.setHeight(mat_a.tileSize(kk).cols());
      b_panel.setWidth(mat_a.tileSize(kk).rows());
    }
    
    if (kk_rank.row() == this_rank.row()) {
      for (SizeType j_local = kk_offset.col(); j_local < distr_a.localNrTiles().cols(); ++j_local) {
        const LocalTileIndex kj_panel(Coord::Col, j_local);
        const LocalTileIndex kj(kk_offset.row(), j_local);

        a_panel.setTile(kj_panel, mat_a.read(kj));
      }
    }
    broadcast(executor_mpi, kk_rank.row(), a_panel, mpi_col_task_chain);

    for (SizeType i_local = distr_b.localNrTiles().rows() - 1; i_local >= 0; --i_local) {
      if (kk_rank.col() == this_rank.col()) {
        auto k_local_col = distr_b.localTileFromGlobalTile<Coord::Col>(k);
	const LocalTileIndex kk_panel(Coord::Col, k_local_col);
        const LocalTileIndex ik(i_local, k_local_col);
	const LocalTileIndex ik_panel(Coord::Row, i_local);

        b_panel.setTile(ik_panel, mat_b.read(ik));
	trmm_B_panel_tile(executor_hp, diag, alpha, a_panel.read(kk_panel), mat_b(ik));
      }
    }
    broadcast(executor_mpi, kk_rank.col(), b_panel, mpi_row_task_chain);

    for (SizeType j_local = bt_offset.col(); j_local < distr_a.localNrTiles().cols(); ++j_local) {
      // Choose queue priority
      auto j = distr_a.globalTileFromLocalTile<Coord::Col>(j_local);
      auto& trailing_executor = (j == k + 1) ? executor_hp : executor_np;
      
      const LocalTileIndex kj_panel(Coord::Col, j_local);
      // Update trailing matrix
      for (SizeType i_local = distr_b.localNrTiles().rows() - 1; i_local >= 0; --i_local) {

        const LocalTileIndex ik_panel(Coord::Row, i_local);
        const LocalTileIndex ij(i_local, j_local);
        gemm_trailing_matrix_tile(trailing_executor, alpha, b_panel.read(ik_panel),
                                  a_panel.read(kj_panel), mat_b(ij));
      }
    }
    
    a_panel.reset();
    b_panel.reset();
  }
}

}
}
}
