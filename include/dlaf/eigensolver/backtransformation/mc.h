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

#include <hpx/include/util.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/common/vector.h"
#include "dlaf/eigensolver/backtransformation/api.h"
#include "dlaf/factorization/qr.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

template <Device device, class T>
void copySingleTile(hpx::shared_future<matrix::Tile<const T, device>> in,
                    hpx::future<matrix::Tile<T, device>> out) {
  hpx::dataflow(dlaf::getCopyExecutor<Device::CPU, Device::CPU>(),
                matrix::unwrapExtendTiles(matrix::copy_o), in, out);
}

template <class Executor, Device device, class T>
void trmmPanel(Executor&& ex, hpx::shared_future<matrix::Tile<const T, device>> t,
               hpx::future<matrix::Tile<T, device>> w) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::trmm_o), blas::Side::Right, blas::Uplo::Upper,
                blas::Op::ConjTrans, blas::Diag::NonUnit, T(1.0), t, w);
}

template <class Executor, Device device, class T>
void gemmUpdateW2(Executor&& ex, hpx::future<matrix::Tile<T, device>> w,
                  hpx::shared_future<matrix::Tile<const T, device>> c,
                  hpx::future<matrix::Tile<T, device>> w2) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::ConjTrans, blas::Op::NoTrans,
                T(1.0), w, c, T(1.0), std::move(w2));
}

template <class Executor, Device device, class T>
void gemmTrailingMatrix(Executor&& ex, hpx::shared_future<matrix::Tile<const T, device>> v,
                        hpx::shared_future<matrix::Tile<const T, device>> w2,
                        hpx::future<matrix::Tile<T, device>> c) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::NoTrans, blas::Op::NoTrans,
                T(-1.0), v, w2, T(1.0), std::move(c));
}

// Implementation based on:
// 1. Algorithm 6 "LAPACK Algorithm for the eigenvector back-transformation", page 15, PhD thesis
// "GPU Accelerated Implementations of a Generalized Eigenvalue Solver for Hermitian Matrices with
// Systematic Energy and Time to Solution Analysis" presented by Raffaele Solcà (2016)
// 2. G. H. Golub and C. F. Van Loan, Matrix Computations, chapter 5, The Johns Hopkins University Press
template <class T>
struct BackTransformation<Backend::MC, Device::CPU, T> {
  static void call_FC(Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v,
                      common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus);
};

template <class T>
void BackTransformation<Backend::MC, Device::CPU, T>::call_FC(
    Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v,
    common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus) {
  using hpx::unwrapping;

  auto executor_hp = dlaf::getHpExecutor<Backend::MC>();
  auto executor_np = dlaf::getNpExecutor<Backend::MC>();

  const SizeType m = mat_c.nrTiles().rows();
  const SizeType n = mat_c.nrTiles().cols();
  const SizeType mb = mat_v.blockSize().rows();

  if (m <= 1 || n == 0)
    return;

  const SizeType nr_reflector = mat_v.size().rows() - mb;

  dlaf::matrix::Distribution dist_t({mb, nr_reflector}, {mb, mb});

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panelsV(n_workspaces,
                                                                        mat_v.distribution());
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panelsW(n_workspaces,
                                                                        mat_v.distribution());
  common::RoundRobin<matrix::Panel<Coord::Row, T, Device::CPU>> panelsW2(n_workspaces,
                                                                         mat_c.distribution());
  common::RoundRobin<matrix::Panel<Coord::Row, T, Device::CPU>> panelsT(n_workspaces, dist_t);

  // Specific for V matrix layout where last column of tiles is empty
  const SizeType num_panel_refls = mat_v.nrTiles().cols() - 1;
  const SizeType last_tile_cols = mat_v.tileSize(GlobalTileIndex(0, m - 1)).cols();

  for (SizeType k = num_panel_refls - 1; k >= 0; --k) {
    bool is_last = (k == num_panel_refls - 1);
    const SizeType t_last_cols = dist_t.tileSize(GlobalTileIndex{k, k}).cols();
    const auto kt = k + 1;

    auto& panelV = panelsV.nextResource();
    auto& panelW = panelsW.nextResource();
    auto& panelW2 = panelsW2.nextResource();
    auto& panelT = panelsT.nextResource();

    panelV.setRangeStart({kt, k});
    panelW.setRangeStart({kt, k});

    for (SizeType i = kt; i < mat_v.nrTiles().rows(); ++i) {
      auto ik = LocalTileIndex{i, k};

      if (i == kt) {
        copySingleTile(mat_v.read(ik), panelV(ik));
        hpx::dataflow(hpx::launch::sync, unwrapping(tile::laset<T>), lapack::MatrixType::Upper, 0.f, 1.f,
                      panelV(ik));
      }
      else {
        hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_v = mat_v.read(ik);

        if (is_last) {
          tile_v = splitTile(tile_v, {{0, 0}, dist_t.tileSize(GlobalTileIndex(k, k))});
        }

        panelV.setTile(ik, mat_v.read(ik));
      }

      // Copy panelV into panelW (panelW will be modified in TRMM operation, while original tile of panelV
      // will be needed in GEMM trailing matrix)
      copySingleTile(panelV.read(ik), panelW(ik));
    }

    const GlobalTileIndex v_start{kt, k};
    auto taus_panel = taus[k];
    const SizeType taupan = (is_last) ? last_tile_cols : mat_v.blockSize().cols();
    auto kk = LocalTileIndex{k, k};
    const LocalTileIndex diag_wp_idx{Coord::Col, k};

    if (is_last) {
      panelT.setHeight(t_last_cols);
      panelW2.setHeight(t_last_cols);
      panelW.setWidth(last_tile_cols);
      panelV.setWidth(last_tile_cols);
    }

    dlaf::factorization::internal::computeTFactor<Backend::MC>(taupan, mat_v, v_start, taus_panel,
                                                               panelT(diag_wp_idx));

    // WH = V T
    hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_t = panelT.read(diag_wp_idx);
    for (const auto& idx : panelW.iteratorLocal()) {
      trmmPanel(executor_np, tile_t, panelW(idx));
    }

    // W2 = W C
    matrix::util::set0(executor_hp, panelW2);
    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{k, j};
      for (SizeType i = kt; i < m; ++i) {
        auto ik = LocalTileIndex{i, k};
        auto ij = LocalTileIndex{i, j};
        gemmUpdateW2(executor_np, panelW(ik), mat_c.read(ij), panelW2(kj));
      }
    }

    // C = C - V W2
    LocalTileIndex c_start{kt, 0};
    LocalTileIndex c_end{m, n};
    for (const auto& idx : iterate_range2d(c_start, c_end)) {
      auto ik = LocalTileIndex{idx.row(), k};
      auto kj = LocalTileIndex{k, idx.col()};
      gemmTrailingMatrix(executor_np, panelV.read(ik), panelW2.read(kj), mat_c(idx));
    }

    panelV.reset();
    panelW.reset();
    panelW2.reset();
    panelT.reset();
  }
}

/// ---- ETI
#define DLAF_EIGENSOLVER_BACKTRANSFORMATION_MC_ETI(KWORD, DATATYPE) \
  KWORD template struct BackTransformation<Backend::MC, Device::CPU, DATATYPE>;

DLAF_EIGENSOLVER_BACKTRANSFORMATION_MC_ETI(extern, float)
DLAF_EIGENSOLVER_BACKTRANSFORMATION_MC_ETI(extern, double)
DLAF_EIGENSOLVER_BACKTRANSFORMATION_MC_ETI(extern, std::complex<float>)
DLAF_EIGENSOLVER_BACKTRANSFORMATION_MC_ETI(extern, std::complex<double>)

}
}
}
