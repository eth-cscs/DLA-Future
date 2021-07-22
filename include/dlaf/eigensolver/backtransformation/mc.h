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
void gemmUpdateW2Start(Executor&& ex, hpx::future<matrix::Tile<T, device>> w,
                       hpx::shared_future<matrix::Tile<const T, device>> c,
                       hpx::future<matrix::Tile<T, device>> w2) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::ConjTrans, blas::Op::NoTrans,
                T(1.0), w, c, T(0.0), std::move(w2));
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

  // Matrix T
  if (m <= 1 || n == 0)
    return;

  const SizeType nr_reflector = mat_v.size().rows() - mb;

  dlaf::matrix::Distribution dist_t({mb, nr_reflector}, {mb, mb});

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panelsVV(n_workspaces,
                                                                         mat_v.distribution());
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panelsW(n_workspaces,
                                                                        mat_v.distribution());
  common::RoundRobin<matrix::Panel<Coord::Row, T, Device::CPU>> panelsW2(n_workspaces,
                                                                         mat_c.distribution());
  common::RoundRobin<matrix::Panel<Coord::Row, T, Device::CPU>> panelsT(n_workspaces, dist_t);

  SizeType last_mb = mat_v.tileSize(GlobalTileIndex(0, m - 1)).cols();

  // Specific for V matrix layout where last column of tiles is empty
  const SizeType last_panel_reflector_idx = mat_v.nrTiles().cols() - 2;

  for (SizeType k = last_panel_reflector_idx; k >= 0; --k) {
    bool is_last = (k == last_panel_reflector_idx) ? true : false;

    auto& panelVV = panelsVV.nextResource();
    auto& panelW = panelsW.nextResource();
    auto& panelW2 = panelsW2.nextResource();
    auto& panelT = panelsT.nextResource();

    for (SizeType i = k + 1; i < mat_v.nrTiles().rows(); ++i) {
      // Copy V panel into VV
      auto ik = LocalTileIndex{i, k};
      auto i_row = LocalTileIndex{Coord::Row, i};
      copySingleTile(mat_v.read(ik), panelVV(i_row));

      // Setting VV
      auto tile_i_row = panelVV(i_row);
      if (i == k + 1) {
        hpx::dataflow(hpx::launch::sync, unwrapping(tile::laset<T>), lapack::MatrixType::Upper, 0.f, 1.f,
                      std::move(tile_i_row));
      }

      // Copy VV into W
      copySingleTile(panelVV.read(i_row), panelW(i_row));
    }

    const GlobalTileIndex v_start{k + 1, k};
    auto taus_panel = taus[k];
    const SizeType taupan = (is_last) ? last_mb : mat_v.blockSize().cols();
    auto kk = LocalTileIndex{k, k};
    const LocalTileIndex diag_wp_idx{Coord::Col, k};

    if (is_last) {
      panelT.setHeight(dist_t.tileSize(GlobalTileIndex{0, k}).cols());
      panelW.setWidth(dist_t.tileSize(GlobalTileIndex{k, k}).cols());
    }

    dlaf::factorization::internal::computeTFactor<Backend::MC>(taupan, mat_v, v_start, taus_panel,
							        panelT(diag_wp_idx));

    // WH = V T
    for (SizeType i = k + 1; i < m; ++i) {
      auto ik = LocalTileIndex{i, k};
      hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_t = panelT.read(diag_wp_idx);

      trmmPanel(executor_np, tile_t, panelW(ik));
    }

    // W2 = W C
    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{Coord::Col, j};
      for (SizeType i = k + 1; i < m; ++i) {
        auto ik = LocalTileIndex{i, k};
        auto ij = LocalTileIndex{i, j};
        panelW.setWidth(mat_v.tileSize(GlobalTileIndex{i, k}).cols());
        if ((i == k + 1)) {
          gemmUpdateW2Start(executor_np, panelW(ik), mat_c.read(ij), panelW2(kj));
        }
        else {
          gemmUpdateW2(executor_np, panelW(ik), mat_c.read(ij), panelW2(kj));
        }
      }
    }

    // C = C - V W2
    for (SizeType i = k + 1; i < m; ++i) {
      auto ik = LocalTileIndex{Coord::Row, i};
      for (SizeType j = 0; j < n; ++j) {
        auto kj = LocalTileIndex{k, j};
        auto ij = LocalTileIndex{i, j};
        gemmTrailingMatrix(executor_np, panelVV.read(ik), panelW2.read(kj), mat_c(ij));
      }
    }

    panelVV.reset(); 
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
