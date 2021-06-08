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
#include "dlaf/common/pipeline.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/communication/init.h"
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
void set_zero(Matrix<T, Device::CPU>& mat) {
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
  using hpx::util::unwrapping;

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
  comm::CommunicatorGrid comm_grid(MPI_COMM_WORLD, 1, 1, common::Ordering::ColumnMajor);
  common::Pipeline<comm::CommunicatorGrid> serial_comm(comm_grid);

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

    for (SizeType i = 0; i < mat_v.nrTiles().rows(); ++i) {
      // Copy V panel into VV
      copySingleTile(mat_v.read(LocalTileIndex(i, k)), mat_vv(LocalTileIndex(i, 0)));

      // Setting VV
      auto setting_vv = unwrapping([=](auto&& tile) {
        if (i <= k) {
          lapack::laset(lapack::MatrixType::General, tile.size().rows(), tile.size().cols(), 0, 0,
                        tile.ptr(), tile.ld());
        }
        else if (i == k + 1) {
          lapack::laset(lapack::MatrixType::Upper, tile.size().rows(), tile.size().cols(), 0, 1,
                        tile.ptr(), tile.ld());
        }
      });
      hpx::dataflow(executor_hp, setting_vv, mat_vv(LocalTileIndex(i, 0)));

      // Copy VV into W
      copySingleTile(mat_vv.read(LocalTileIndex(i, 0)), mat_w(LocalTileIndex(i, 0)));
    }

    // Reset W2 to zero
    matrix::util::set(mat_w2, [](auto&&) { return 0; });

    const GlobalTileIndex v_start{k + 1, k};
    auto taus_panel = taus[k];
    int taupan = (is_last) ? last_mb : mat_v.blockSize().cols();
    dlaf::factorization::internal::computeTFactor<Backend::MC>(taupan, mat_v, v_start, taus_panel,
                                                               mat_t(LocalTileIndex{k, k}), serial_comm);

    for (SizeType i = k + 1; i < m; ++i) {
      auto kk = LocalTileIndex{k, k};
      // WH = V T
      auto ik = LocalTileIndex{i, 0};
      trmmPanel(executor_np, mat_t.read(kk), mat_w(ik));
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
