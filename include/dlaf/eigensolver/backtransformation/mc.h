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
#include "dlaf/common/vector.h"
#include "dlaf/eigensolver/backtransformation/api.h"
#include "dlaf/factorization/qr.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

template <class T>
void setZero(Matrix<T, Device::CPU>& mat) {
  dlaf::matrix::util::set(mat, [](auto&&) { return static_cast<T>(0.0); });
}

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
// 1. Part of algorithm 6 "LAPACK Algorithm for the eigenvector back-transformation", page 15, PhD thesis
// "GPU Accelerated Implementations of a Generalized Eigenvalue Solver for Hermitian Matrices with
// Systematic Energy and Time to Solution Analysis" presented by Raffaele Solcà (2016)
// 2. Report "Gep + back-transformation", Alberto Invernizzi (2020)
// 3. Report "Reduction to band + back-transformation", Raffaele Solcà (2020)
// 4. G. H. Golub and C. F. Van Loan, Matrix Computations, chapter 5, The Johns Hopkins University Press
template <class T>
struct BackTransformation<Backend::MC, Device::CPU, T> {
  static void call_FC(Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v,
                      common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus);
  static void call_FC(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_c,
                      Matrix<const T, Device::CPU>& mat_v,
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
  const SizeType mv = mat_v.nrTiles().rows();
  const SizeType nv = mat_v.nrTiles().cols();
  const SizeType mb = mat_v.blockSize().rows();
  const SizeType nb = mat_v.blockSize().cols();
  const SizeType ms = mat_v.size().rows();
  const SizeType ns = mat_v.size().cols();

  // Matrix T
  int tottaus;
  if (ms < mb || ms == 0 || nv == 0)
    tottaus = 0;
  else
    tottaus = (ms / mb - 1) * mb + ms % mb;

  if (tottaus == 0)
    return;

  LocalElementSize sizeT(tottaus, tottaus);
  TileElementSize blockSizeT(mb, mb);
  Matrix<T, Device::CPU> mat_t(sizeT, blockSizeT);

  Matrix<T, Device::CPU> mat_vv({mat_v.size().rows(), mb}, mat_v.blockSize());
  Matrix<T, Device::CPU> mat_w({mat_v.size().rows(), mb}, mat_v.blockSize());
  Matrix<T, Device::CPU> mat_w2({mb, mat_c.size().cols()}, mat_c.blockSize());

  SizeType last_mb;
  if (mat_v.blockSize().cols() == 1) {
    last_mb = 1;
  }
  else if (mat_v.size().cols()) {
    if (mat_v.size().cols() % mat_v.blockSize().cols() == 0)
      last_mb = mat_v.blockSize().cols();
    else
      last_mb = mat_v.size().cols() % mat_v.blockSize().cols();
  }

  // Specific for V matrix layout where last column of tiles is empty
  const SizeType last_reflector_idx = mat_v.nrTiles().cols() - 2;

  for (SizeType k = last_reflector_idx; k >= 0; --k) {
    bool is_last = (k == last_reflector_idx) ? true : false;

    for (SizeType i = k + 1; i < mat_v.nrTiles().rows(); ++i) {
      // Copy V panel into VV
      copySingleTile(mat_v.read(LocalTileIndex(i, k)), mat_vv(LocalTileIndex(i, 0)));

      // Setting VV
      auto tile_i0 = mat_vv(LocalTileIndex(i, 0));
      if (i <= k) {
        hpx::dataflow(hpx::launch::sync, unwrapping(tile::set0<T>), std::move(tile_i0));
      }
      else if (i == k + 1) {
        hpx::dataflow(hpx::launch::sync, unwrapping(tile::laset<T>), lapack::MatrixType::Upper, 0.f, 1.f,
                      std::move(tile_i0));
      }

      // Copy VV into W
      copySingleTile(mat_vv.read(LocalTileIndex(i, 0)), mat_w(LocalTileIndex(i, 0)));
    }

    // Reset W2 to zero
    setZero(mat_w2);

    // TODO: instead of using a full matrix, choose a "column" matrix. The problem is that last tile
    // should be square but may have different size.
    const GlobalTileIndex v_start{k + 1, k};
    auto taus_panel = taus[k];
    const SizeType taupan = (is_last) ? last_mb : mat_v.blockSize().cols();
    dlaf::factorization::internal::computeTFactor<Backend::MC>(taupan, mat_v, v_start, taus_panel,
                                                               mat_t(LocalTileIndex{k, k}));

    for (SizeType i = k + 1; i < m; ++i) {
      auto kk = LocalTileIndex{k, k};
      // WH = V T
      auto ik = LocalTileIndex{i, 0};
      hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_t = mat_t.read(kk);
      auto tile_w = mat_w(ik);

      if (mat_t.tileSize(GlobalTileIndex{k, k}).rows() != mat_w.tileSize(GlobalTileIndex{i, 0}).cols()) {
        TileElementIndex origin(0, 0);
        TileElementSize size(mat_t.tileSize(GlobalTileIndex{k, k}).rows(),
                             mat_t.tileSize(GlobalTileIndex{k, k}).cols());
        const matrix::SubTileSpec spec({origin, size});
        auto subtile_w = splitTile(tile_w, spec);
        trmmPanel(executor_np, tile_t, std::move(subtile_w));
      }
      else
        trmmPanel(executor_np, mat_t.read(kk), std::move(tile_w));
    }

    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{0, j};
      for (SizeType i = k + 1; i < m; ++i) {
        auto ik = LocalTileIndex{i, 0};
        auto ij = LocalTileIndex{i, j};
        // W2 = W C
        gemmUpdateW2(executor_np, mat_w(ik), mat_c.read(ij), mat_w2(kj));
      }
    }

    for (SizeType i = k + 1; i < m; ++i) {
      auto ik = LocalTileIndex{i, 0};
      for (SizeType j = 0; j < n; ++j) {
        auto kj = LocalTileIndex{0, j};
        auto ij = LocalTileIndex{i, j};
        // C = C - V W2
        gemmTrailingMatrix(executor_np, mat_vv.read(ik), mat_w2.read(kj), mat_c(ij));
      }
    }
  }
}

template <class T>
void BackTransformation<Backend::MC, Device::CPU, T>::call_FC(
    comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v,
    common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus) {
  DLAF_UNIMPLEMENTED(grid);
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
