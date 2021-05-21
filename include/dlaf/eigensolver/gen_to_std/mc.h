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

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/eigensolver/gen_to_std/api.h"
#include "dlaf/executors.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

template <class T>
void hegst_diag_tile(hpx::execution::parallel_executor executor_hp,
                     hpx::future<matrix::Tile<T, Device::CPU>> a_kk,
                     hpx::future<matrix::Tile<T, Device::CPU>> l_kk) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::hegst_o), 1, blas::Uplo::Lower,
                std::move(a_kk), std::move(l_kk));
}

template <class T>
void trsm_panel_tile(hpx::execution::parallel_executor executor_hp,
                     hpx::shared_future<matrix::Tile<const T, Device::CPU>> l_kk,
                     hpx::future<matrix::Tile<T, Device::CPU>> a_ik) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), blas::Side::Right,
                blas::Uplo::Lower, blas::Op::ConjTrans, blas::Diag::NonUnit, T(1.0), l_kk,
                std::move(a_ik));
}

template <class T>
void hemm_panel_tile(hpx::execution::parallel_executor executor_hp,
                     hpx::shared_future<matrix::Tile<const T, Device::CPU>> a_kk,
                     hpx::shared_future<matrix::Tile<const T, Device::CPU>> l_ik,
                     hpx::future<matrix::Tile<T, Device::CPU>> a_ik) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::hemm_o), blas::Side::Right,
                blas::Uplo::Lower, T(-0.5), a_kk, l_ik, T(1.0), std::move(a_ik));
}

template <class T>
void her2k_trailing_diag_tile(hpx::execution::parallel_executor ex,
                              hpx::shared_future<matrix::Tile<const T, Device::CPU>> a_jk,
                              hpx::shared_future<matrix::Tile<const T, Device::CPU>> l_jk,
                              hpx::future<matrix::Tile<T, Device::CPU>> a_kk) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::her2k_o), blas::Uplo::Lower, blas::Op::NoTrans,
                T(-1.0), a_jk, l_jk, BaseType<T>(1.0), std::move(a_kk));
}

template <class T>
void gemm_trailing_matrix_tile(hpx::execution::parallel_executor ex,
                               hpx::shared_future<matrix::Tile<const T, Device::CPU>> mat_ik,
                               hpx::shared_future<matrix::Tile<const T, Device::CPU>> mat_jk,
                               hpx::future<matrix::Tile<T, Device::CPU>> a_ij) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::NoTrans, blas::Op::ConjTrans,
                T(-1.0), mat_ik, mat_jk, T(1.0), std::move(a_ij));
}

template <class T>
void trsm_panel_update_tile(hpx::execution::parallel_executor executor_hp,
                            hpx::shared_future<matrix::Tile<const T, Device::CPU>> l_jj,
                            hpx::future<matrix::Tile<T, Device::CPU>> a_jk) {
  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::trsm_o), blas::Side::Left,
                blas::Uplo::Lower, blas::Op::NoTrans, blas::Diag::NonUnit, T(1.0), l_jj,
                std::move(a_jk));
}

template <class T>
void gemm_panel_update_tile(hpx::execution::parallel_executor ex,
                            hpx::shared_future<matrix::Tile<const T, Device::CPU>> l_ij,
                            hpx::shared_future<matrix::Tile<const T, Device::CPU>> a_jk,
                            hpx::future<matrix::Tile<T, Device::CPU>> a_ik) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::NoTrans, blas::Op::NoTrans,
                T(-1.0), l_ij, a_jk, T(1.0), std::move(a_ik));
}

// Implementation based on LAPACK Algorithm for the transformation from generalized to standard
// eigenproblem (xHEGST)
template <class T>
struct GenToStd<Backend::MC, Device::CPU, T> {
  static void call_L(Matrix<T, Device::CPU>& mat_a, Matrix<T, Device::CPU>& mat_l);
};

template <class T>
void GenToStd<Backend::MC, Device::CPU, T>::call_L(Matrix<T, Device::CPU>& mat_a,
                                                   Matrix<T, Device::CPU>& mat_l) {
  auto executor_hp = dlaf::getHpExecutor<Backend::MC>();
  auto executor_np = dlaf::getNpExecutor<Backend::MC>();

  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().cols();

  for (SizeType k = 0; k < nrtile; ++k) {
    const auto kk = LocalTileIndex{k, k};

    // Direct transformation to standard eigenvalue problem of the diagonal tile
    hegst_diag_tile(executor_hp, mat_a(kk), mat_l(kk));

    const LocalTileIndex ai_start(k + 1, k);
    const LocalTileIndex ai_end(nrtile, k + 1);
    auto ai_panel = common::iterate_range2d(ai_start, ai_end);

    for (const auto& ik : ai_panel) {
      trsm_panel_tile(executor_hp, mat_l.read(kk), mat_a(ik));
      hemm_panel_tile(executor_hp, mat_a.read(kk), mat_l.read(ik), mat_a(ik));
    }

    const LocalTileIndex ti_start(k + 1, k + 1);
    const LocalTileIndex ti_end(nrtile, nrtile);
    for (const auto& ij : common::iterate_range2d(ti_start, ti_end)) {
      const auto jk = LocalTileIndex{ij.col(), k};
      const auto ik = LocalTileIndex{ij.row(), k};
      auto& trailing_matrix_executor = (ij.col() == k + 1) ? executor_hp : executor_np;

      if (ij.row() == ij.col()) {
        her2k_trailing_diag_tile(trailing_matrix_executor, mat_a.read(ik), mat_l.read(jk), mat_a(ij));
      }
      else if (ij.row() > ij.col()) {
        gemm_trailing_matrix_tile(trailing_matrix_executor, mat_a.read(ik), mat_l.read(jk), mat_a(ij));
        gemm_trailing_matrix_tile(trailing_matrix_executor, mat_l.read(ik), mat_a.read(jk), mat_a(ij));
      }
    }

    for (const auto& ik : ai_panel) {
      hemm_panel_tile(executor_np, mat_a.read(kk), mat_l.read(ik), mat_a(ik));
    }

    for (SizeType j = k + 1; j < nrtile; ++j) {
      trsm_panel_update_tile(executor_hp, mat_l.read(LocalTileIndex{j, j}), mat_a(LocalTileIndex{j, k}));

      for (SizeType i = j + 1; i < nrtile; ++i) {
        gemm_panel_update_tile(executor_np, mat_l.read(LocalTileIndex{i, j}),
                               mat_a.read(LocalTileIndex{j, k}), mat_a(LocalTileIndex{i, k}));
      }
    }
  }
}

/// ---- ETI
#define DLAF_GENTOSTD_MC_ETI(KWORD, DATATYPE) \
  KWORD template struct GenToStd<Backend::MC, Device::CPU, DATATYPE>;

DLAF_GENTOSTD_MC_ETI(extern, float)
DLAF_GENTOSTD_MC_ETI(extern, double)
DLAF_GENTOSTD_MC_ETI(extern, std::complex<float>)
DLAF_GENTOSTD_MC_ETI(extern, std::complex<double>)
}
}
}
