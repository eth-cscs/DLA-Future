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
        gemm_trailing_matrix_tile(executor_np, alpha, mat_a.read(LocalTileIndex{i, k}),
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
        gemm_trailing_matrix_tile(executor_np, op, alpha, mat_a.read(LocalTileIndex{k, i}),
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

      for (SizeType i = 0; i < k; ++i) {
        gemm_trailing_matrix_tile(executor_np, alpha, mat_a.read(LocalTileIndex{i, k}),
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
        gemm_trailing_matrix_tile(executor_np, op, alpha, mat_a.read(LocalTileIndex{k, i}),
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
        gemm_trailing_matrix_tile(executor_np, alpha, mat_b.read(ik),
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
        gemm_trailing_matrix_tile(executor_np, op, alpha, mat_b.read(ik),
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
        gemm_trailing_matrix_tile(executor_np, alpha, mat_b.read(ik),
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
        gemm_trailing_matrix_tile(executor_np, op, alpha, mat_b.read(ik),
                                  mat_a.read(LocalTileIndex{j, k}), mat_b(LocalTileIndex{i, j}));
      }

      trmm_B_panel_tile(executor_hp, op, diag, alpha, mat_a.read(LocalTileIndex{k, k}), mat_b(ik));
    }
  }
}

}
}
}
