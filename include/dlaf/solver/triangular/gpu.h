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

#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/executors.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/solver/triangular/api.h"
#include "dlaf/tile.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace solver {
namespace internal {

template <class T>
struct Triangular<Backend::GPU, Device::GPU, T> {
  static void call_LLN(blas::Diag diag, T alpha, Matrix<const T, Device::GPU>& mat_a,
                       Matrix<T, Device::GPU>& mat_b);
  static void call_LLT(blas::Op op, blas::Diag diag, T alpha, Matrix<const T, Device::GPU>& mat_a,
                       Matrix<T, Device::GPU>& mat_b);
  static void call_LUN(blas::Diag diag, T alpha, Matrix<const T, Device::GPU>& mat_a,
                       Matrix<T, Device::GPU>& mat_b);
  static void call_LUT(blas::Op op, blas::Diag diag, T alpha, Matrix<const T, Device::GPU>& mat_a,
                       Matrix<T, Device::GPU>& mat_b);
  static void call_RLN(blas::Diag diag, T alpha, Matrix<const T, Device::GPU>& mat_a,
                       Matrix<T, Device::GPU>& mat_b);
  static void call_RLT(blas::Op op, blas::Diag diag, T alpha, Matrix<const T, Device::GPU>& mat_a,
                       Matrix<T, Device::GPU>& mat_b);
  static void call_RUN(blas::Diag diag, T alpha, Matrix<const T, Device::GPU>& mat_a,
                       Matrix<T, Device::GPU>& mat_b);
  static void call_RUT(blas::Op op, blas::Diag diag, T alpha, Matrix<const T, Device::GPU>& mat_a,
                       Matrix<T, Device::GPU>& mat_b);
  static void call_LLN(comm::CommunicatorGrid grid, blas::Diag diag, T alpha,
                       Matrix<const T, Device::GPU>& mat_a, Matrix<T, Device::GPU>& mat_b);
};

namespace lln {
template <class Executor, class T>
void trsm_B_panel_tile(Executor&& ex, blas::Diag diag, T alpha,
                       hpx::shared_future<matrix::Tile<const T, Device::GPU>> in_tile,
                       hpx::future<matrix::Tile<T, Device::GPU>> out_tile) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(tile::trsm_o), blas::Side::Left,
                blas::Uplo::Lower, blas::Op::NoTrans, diag, alpha, std::move(in_tile),
                std::move(out_tile));
}

template <class Executor, class T>
void gemm_trailing_matrix_tile(Executor&& ex, T beta,
                               hpx::shared_future<matrix::Tile<const T, Device::GPU>> a_tile,
                               hpx::shared_future<matrix::Tile<const T, Device::GPU>> b_tile,
                               hpx::future<matrix::Tile<T, Device::GPU>> c_tile) {
  hpx::dataflow(std::forward<Executor>(ex), matrix::unwrapExtendTiles(tile::gemm_o),
                blas::Op::NoTrans, blas::Op::NoTrans, beta, std::move(a_tile), std::move(b_tile), T(1.0),
                std::move(c_tile));
}
}

template <class T>
void Triangular<Backend::GPU, Device::GPU, T>::call_LLN(blas::Diag diag, T alpha,
                                                       Matrix<const T, Device::GPU>& mat_a,
                                                       Matrix<T, Device::GPU>& mat_b) {
  auto executor_hp = dlaf::getHpExecutor<Backend::GPU>();
  auto executor_np = dlaf::getNpExecutor<Backend::GPU>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < m; ++k) {
    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{k, j};

      // Triangular solve of k-th row Panel of B
      lln::trsm_B_panel_tile(executor_hp, diag, alpha, mat_a.read(LocalTileIndex{k, k}), mat_b(kj));

      for (SizeType i = k + 1; i < m; ++i) {
        // Choose queue priority
        auto& trailing_executor = (i == k + 1) ? executor_hp : executor_np;
        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        lln::gemm_trailing_matrix_tile(trailing_executor, beta, mat_a.read(LocalTileIndex{i, k}),
                                       mat_b.read(kj), mat_b(LocalTileIndex{i, j}));
      }
    }
  }
}

template <class T>
void Triangular<Backend::GPU, Device::GPU, T>::call_LLT(blas::Op op, blas::Diag diag, T alpha,
                                                       Matrix<const T, Device::GPU>& mat_a,
                                                       Matrix<T, Device::GPU>& mat_b) {
  constexpr auto Left = blas::Side::Left;
  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto NoTrans = blas::Op::NoTrans;

  auto executor_hp = dlaf::getHpExecutor<Backend::GPU>();
  auto executor_np = dlaf::getNpExecutor<Backend::GPU>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = m - 1; k > -1; --k) {
    for (SizeType j = n - 1; j > -1; --j) {
      auto kj = LocalTileIndex{k, j};
      // Triangular solve of k-th row Panel of B
      hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), Left, Lower, op, diag, alpha,
                    mat_a.read(LocalTileIndex{k, k}), std::move(mat_b(kj)));

      for (SizeType i = k - 1; i > -1; --i) {
        // Choose queue priority
        auto& trailing_executor = (i == k - 1) ? executor_hp : executor_np;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        hpx::dataflow(trailing_executor, matrix::unwrapExtendTiles(tile::gemm_o), op, NoTrans, beta,
                      mat_a.read(LocalTileIndex{k, i}), mat_b.read(kj), T(1.0),
                      std::move(mat_b(LocalTileIndex{i, j})));
      }
    }
  }
}

template <class T>
void Triangular<Backend::GPU, Device::GPU, T>::call_LUN(blas::Diag diag, T alpha,
                                                       Matrix<const T, Device::GPU>& mat_a,
                                                       Matrix<T, Device::GPU>& mat_b) {
  constexpr auto Left = blas::Side::Left;
  constexpr auto Upper = blas::Uplo::Upper;
  constexpr auto NoTrans = blas::Op::NoTrans;

  auto executor_hp = dlaf::getHpExecutor<Backend::GPU>();
  auto executor_np = dlaf::getNpExecutor<Backend::GPU>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = m - 1; k > -1; --k) {
    for (SizeType j = n - 1; j > -1; --j) {
      auto kj = LocalTileIndex{k, j};
      // Triangular solve of k-th row Panel of B
      hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), Left, Upper, NoTrans, diag,
                    alpha, mat_a.read(LocalTileIndex{k, k}), std::move(mat_b(kj)));

      for (SizeType i = k - 1; i > -1; --i) {
        // Choose queue priority
        auto& trailing_executor = (i == k - 1) ? executor_hp : executor_np;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        hpx::dataflow(trailing_executor, matrix::unwrapExtendTiles(tile::gemm_o), NoTrans, NoTrans, beta,
                      mat_a.read(LocalTileIndex{i, k}), mat_b.read(kj), T(1.0),
                      std::move(mat_b(LocalTileIndex{i, j})));
      }
    }
  }
}

template <class T>
void Triangular<Backend::GPU, Device::GPU, T>::call_LUT(blas::Op op, blas::Diag diag, T alpha,
                                                       Matrix<const T, Device::GPU>& mat_a,
                                                       Matrix<T, Device::GPU>& mat_b) {
  constexpr auto Left = blas::Side::Left;
  constexpr auto Upper = blas::Uplo::Upper;
  constexpr auto NoTrans = blas::Op::NoTrans;

  auto executor_hp = dlaf::getHpExecutor<Backend::GPU>();
  auto executor_np = dlaf::getNpExecutor<Backend::GPU>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < m; ++k) {
    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{k, j};

      // Triangular solve of k-th row Panel of B
      hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), Left, Upper, op, diag, alpha,
                    mat_a.read(LocalTileIndex{k, k}), std::move(mat_b(kj)));

      for (SizeType i = k + 1; i < m; ++i) {
        // Choose queue priority
        auto& trailing_executor = (i == k + 1) ? executor_hp : executor_np;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        hpx::dataflow(trailing_executor, matrix::unwrapExtendTiles(tile::gemm_o), op, NoTrans, beta,
                      mat_a.read(LocalTileIndex{k, i}), mat_b.read(kj), T(1.0),
                      std::move(mat_b(LocalTileIndex{i, j})));
      }
    }
  }
}

template <class T>
void Triangular<Backend::GPU, Device::GPU, T>::call_RLN(blas::Diag diag, T alpha,
                                                       Matrix<const T, Device::GPU>& mat_a,
                                                       Matrix<T, Device::GPU>& mat_b) {
  constexpr auto Right = blas::Side::Right;
  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto NoTrans = blas::Op::NoTrans;

  auto executor_hp = dlaf::getHpExecutor<Backend::GPU>();
  auto executor_np = dlaf::getNpExecutor<Backend::GPU>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = n - 1; k > -1; --k) {
    for (SizeType i = m - 1; i > -1; --i) {
      auto ik = LocalTileIndex{i, k};

      // Triangular solve of k-th col Panel of B
      hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), Right, Lower, NoTrans, diag,
                    alpha, mat_a.read(LocalTileIndex{k, k}), std::move(mat_b(ik)));

      for (SizeType j = k - 1; j > -1; --j) {
        // Choose queue priority
        auto& trailing_executor = (j == k - 1) ? executor_hp : executor_np;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        hpx::dataflow(trailing_executor, matrix::unwrapExtendTiles(tile::gemm_o), NoTrans, NoTrans, beta,
                      mat_b.read(ik), mat_a.read(LocalTileIndex{k, j}), T(1.0),
                      std::move(mat_b(LocalTileIndex{i, j})));
      }
    }
  }
}

template <class T>
void Triangular<Backend::GPU, Device::GPU, T>::call_RLT(blas::Op op, blas::Diag diag, T alpha,
                                                       Matrix<const T, Device::GPU>& mat_a,
                                                       Matrix<T, Device::GPU>& mat_b) {
  constexpr auto Right = blas::Side::Right;
  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto NoTrans = blas::Op::NoTrans;

  auto executor_hp = dlaf::getHpExecutor<Backend::GPU>();
  auto executor_np = dlaf::getNpExecutor<Backend::GPU>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < n; ++k) {
    for (SizeType i = 0; i < m; ++i) {
      auto ik = LocalTileIndex{i, k};

      // Triangular solve of k-th col Panel of B
      hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), Right, Lower, op, diag, alpha,
                    mat_a.read(LocalTileIndex{k, k}), std::move(mat_b(ik)));

      for (SizeType j = k + 1; j < n; ++j) {
        // Choose queue priority
        auto& trailing_executor = (j == k + 1) ? executor_hp : executor_np;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        hpx::dataflow(trailing_executor, matrix::unwrapExtendTiles(tile::gemm_o), NoTrans, op, beta,
                      mat_b.read(ik), mat_a.read(LocalTileIndex{j, k}), T(1.0),
                      std::move(mat_b(LocalTileIndex{i, j})));
      }
    }
  }
}

template <class T>
void Triangular<Backend::GPU, Device::GPU, T>::call_RUN(blas::Diag diag, T alpha,
                                                       Matrix<const T, Device::GPU>& mat_a,
                                                       Matrix<T, Device::GPU>& mat_b) {
  constexpr auto Right = blas::Side::Right;
  constexpr auto Upper = blas::Uplo::Upper;
  constexpr auto NoTrans = blas::Op::NoTrans;

  auto executor_hp = dlaf::getHpExecutor<Backend::GPU>();
  auto executor_np = dlaf::getNpExecutor<Backend::GPU>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < n; ++k) {
    for (SizeType i = 0; i < m; ++i) {
      auto ik = LocalTileIndex{i, k};

      // Triangular solve of k-th col Panel of B
      hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), Right, Upper, NoTrans, diag,
                    alpha, mat_a.read(LocalTileIndex{k, k}), std::move(mat_b(ik)));

      for (SizeType j = k + 1; j < n; ++j) {
        // Choose queue priority
        auto& trailing_executor = (j == k + 1) ? executor_hp : executor_np;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        hpx::dataflow(trailing_executor, matrix::unwrapExtendTiles(tile::gemm_o), NoTrans, NoTrans, beta,
                      mat_b.read(ik), mat_a.read(LocalTileIndex{k, j}), T(1.0),
                      std::move(mat_b(LocalTileIndex{i, j})));
      }
    }
  }
}

template <class T>
void Triangular<Backend::GPU, Device::GPU, T>::call_RUT(blas::Op op, blas::Diag diag, T alpha,
                                                       Matrix<const T, Device::GPU>& mat_a,
                                                       Matrix<T, Device::GPU>& mat_b) {
  constexpr auto Right = blas::Side::Right;
  constexpr auto Upper = blas::Uplo::Upper;
  constexpr auto NoTrans = blas::Op::NoTrans;

  auto executor_hp = dlaf::getHpExecutor<Backend::GPU>();
  auto executor_np = dlaf::getNpExecutor<Backend::GPU>();

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = n - 1; k > -1; --k) {
    for (SizeType i = m - 1; i > -1; --i) {
      auto ik = LocalTileIndex{i, k};

      // Triangular solve of k-th col Panel of B
      hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), Right, Upper, op, diag, alpha,
                    mat_a.read(LocalTileIndex{k, k}), std::move(mat_b(ik)));

      for (SizeType j = k - 1; j > -1; --j) {
        // Choose queue priority
        auto& trailing_executor = (j == k - 1) ? executor_hp : executor_np;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        hpx::dataflow(trailing_executor, matrix::unwrapExtendTiles(tile::gemm_o), NoTrans, op, beta,
                      mat_b.read(ik), mat_a.read(LocalTileIndex{j, k}), T(1.0),
                      std::move(mat_b(LocalTileIndex{i, j})));
      }
    }
  }
}

template <class T>
void Triangular<Backend::GPU, Device::GPU, T>::call_LLN(comm::CommunicatorGrid grid, blas::Diag diag,
                                                       T alpha, Matrix<const T, Device::GPU>& mat_a,
                                                       Matrix<T, Device::GPU>& mat_b) {
  using common::internal::vector;
  using ConstTileType = typename Matrix<T, Device::GPU>::ConstTileType;

  auto executor_hp = dlaf::getHpExecutor<Backend::GPU>();
  auto executor_np = dlaf::getNpExecutor<Backend::GPU>();
  auto executor_mpi = dlaf::getMPIExecutor<Backend::GPU>();

  common::Pipeline<comm::CommunicatorGrid> serial_comm(std::move(grid));

  const matrix::Distribution& distr_a = mat_a.distribution();
  const matrix::Distribution& distr_b = mat_b.distribution();
  SizeType a_rows = mat_a.nrTiles().rows();
  auto a_local_rows = distr_a.localNrTiles().rows();
  auto b_local_cols = distr_b.localNrTiles().cols();

  for (SizeType k = 0; k < a_rows; ++k) {
    // Create a placeholder that will store the shared futures representing the panel
    vector<hpx::shared_future<ConstTileType>> panel(distr_b.localNrTiles().cols());

    auto k_rank_row = distr_a.rankGlobalTile<Coord::Row>(k);
    auto k_rank_col = distr_a.rankGlobalTile<Coord::Col>(k);

    hpx::shared_future<ConstTileType> kk_tile;

    if (mat_a.rankIndex().row() == k_rank_row) {
      auto k_local_row = distr_a.localTileFromGlobalTile<Coord::Row>(k);

      if (mat_a.rankIndex().col() == k_rank_col) {
        // Broadcast A(kk) row-wise
        auto k_local_col = distr_a.localTileFromGlobalTile<Coord::Col>(k);
        auto kk = LocalTileIndex{k_local_row, k_local_col};

        kk_tile = mat_a.read(kk);
        comm::send_tile(executor_mpi, serial_comm, Coord::Row, mat_a.read(kk));
      }
      else {
        kk_tile = comm::recv_tile<T, Device::GPU>(executor_mpi, serial_comm, Coord::Row,
                                     mat_a.tileSize(GlobalTileIndex(k, k)), k_rank_col);
      }
    }

    for (SizeType j_local = 0; j_local < b_local_cols; ++j_local) {
      auto j = distr_b.globalTileFromLocalTile<Coord::Col>(j_local);

      // Triangular solve B's k-th row panel and broadcast B(kj) column-wise
      if (mat_b.rankIndex().row() == k_rank_row) {
        auto k_local_row = distr_b.localTileFromGlobalTile<Coord::Row>(k);
        auto kj = LocalTileIndex{k_local_row, j_local};
        lln::trsm_B_panel_tile(executor_hp, diag, alpha, kk_tile, mat_b(kj));
        panel[j_local] = mat_b.read(kj);
        if (k != (mat_b.nrTiles().rows() - 1)) {
          comm::send_tile(executor_mpi, serial_comm, Coord::Col, panel[j_local]);
        }
      }
      else {
        if (k != (mat_b.nrTiles().rows() - 1)) {
          panel[j_local] = comm::recv_tile<T, Device::GPU>(executor_mpi, serial_comm, Coord::Col,
                                              mat_b.tileSize(GlobalTileIndex(k, j)), k_rank_row);
        }
      }
    }

    for (SizeType i_local = distr_a.nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
         i_local < a_local_rows; ++i_local) {
      auto i = distr_a.globalTileFromLocalTile<Coord::Row>(i_local);

      // Choose queue priority
      auto& trailing_executor = (i == k + 1) ? executor_hp : executor_np;

      hpx::shared_future<ConstTileType> ik_tile;

      // Broadcast A(ik) row-wise
      if (mat_a.rankIndex().col() == k_rank_col) {
        auto k_local_col = distr_a.localTileFromGlobalTile<Coord::Col>(k);
        auto ik = LocalTileIndex{i_local, k_local_col};

        ik_tile = mat_a.read(ik);
        comm::send_tile(executor_mpi, serial_comm, Coord::Row, mat_a.read(ik));
      }
      else {
        ik_tile = comm::recv_tile<T, Device::GPU>(executor_mpi, serial_comm, Coord::Row,
                                     mat_a.tileSize(GlobalTileIndex(i, k)), k_rank_col);
      }

      // Update trailing matrix
      for (SizeType j_local = 0; j_local < b_local_cols; ++j_local) {
        T beta = T(-1.0) / alpha;
        lln::gemm_trailing_matrix_tile(trailing_executor, beta, ik_tile, panel[j_local],
                                       mat_b(LocalTileIndex{i_local, j_local}));
      }
    }
  }
}

/// ---- ETI
#define DLAF_SOLVER_TRIANGULAR_GPU_ETI(KWORD, DATATYPE) \
  KWORD template struct Triangular<Backend::GPU, Device::GPU, DATATYPE>;

DLAF_SOLVER_TRIANGULAR_GPU_ETI(extern, float)
DLAF_SOLVER_TRIANGULAR_GPU_ETI(extern, double)
DLAF_SOLVER_TRIANGULAR_GPU_ETI(extern, std::complex<float>)
DLAF_SOLVER_TRIANGULAR_GPU_ETI(extern, std::complex<double>)
}
}
}
