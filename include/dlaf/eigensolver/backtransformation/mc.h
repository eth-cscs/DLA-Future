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
#include <hpx/include/threads.hpp>
#include <hpx/include/util.hpp>
#include <hpx/local/future.hpp>

#include <blas.hh>

#include "dlaf/eigensolver/backtransformation/api.h"

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/communication/init.h"
#include "dlaf/factorization/qr.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace solver {
namespace internal {

using namespace dlaf::matrix;

template <class T>
void set_zero(Matrix<T, Device::CPU>& mat) {
  dlaf::matrix::util::set(mat, [](auto&&) { return static_cast<T>(0.0); });
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
  constexpr auto Left = blas::Side::Left;
  constexpr auto Right = blas::Side::Right;
  constexpr auto Upper = blas::Uplo::Upper;
  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto ConjTrans = blas::Op::ConjTrans;
  constexpr auto NonUnit = blas::Diag::NonUnit;

  using hpx::util::unwrapping;

  using hpx::execution::parallel_executor;
  using hpx::resource::get_thread_pool;
  using hpx::threads::thread_priority;

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
      auto copy_v_into_vv = unwrapping([=](auto&& tile_v, auto&& tile_vv, TileElementSize region) {
        void (&cpy)(const matrix::Tile<const T, Device::CPU>&, const matrix::Tile<T, Device::CPU>&) =
            copy<T>;
        cpy(tile_v, tile_vv);
      });
      hpx::dataflow(executor_hp, copy_v_into_vv, mat_v.read(LocalTileIndex(i, k)),
                    mat_vv(LocalTileIndex(i, 0)), mat_vv.tileSize(GlobalTileIndex(i, 0)));

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
      auto copy_vv_into_w = unwrapping([=](auto&& tile_vv, auto&& tile_w) {
        void (&cpy)(const matrix::Tile<const T, Device::CPU>&, const matrix::Tile<T, Device::CPU>&) =
            copy<T>;
        cpy(tile_vv, tile_w);
      });
      hpx::dataflow(executor_hp, copy_vv_into_w, mat_vv.read(LocalTileIndex(i, 0)),
                    mat_w(LocalTileIndex(i, 0)));
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
      //hpx::dataflow(executor_np, hpx::util::unwrapping(tile::trmm<T, Device::CPU>), Right, Upper, ConjTrans, NonUnit, 1.0, mat_t.read(kk), std::move(mat_w(ik)));
      hpx::dataflow(executor_np, matrix::unwrapExtendTiles(tile::trmm_o), Right, Upper, ConjTrans, NonUnit, T(1.0), mat_t.read(kk), std::move(mat_w(ik)));
    }

    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{0, j};
      for (SizeType i = k + 1; i < m; ++i) {
        auto ik = LocalTileIndex{i, 0};
        auto ij = LocalTileIndex{i, j};
        // W2 = W C
	//        hpx::dataflow(executor_np, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans, NoTrans, 1.0, std::move(mat_w(ik)), mat_c.read(ij), 1.0, std::move(mat_w2(kj)));
	hpx::dataflow(executor_np, matrix::unwrapExtendTiles(tile::gemm_o), ConjTrans, NoTrans, T(1.0), std::move(mat_w(ik)), mat_c.read(ij), T(1.0), std::move(mat_w2(kj)));
      }
    }

    for (SizeType i = k + 1; i < m; ++i) {
      auto ik = LocalTileIndex{i, 0};
      for (SizeType j = 0; j < n; ++j) {
        auto kj = LocalTileIndex{0, j};
        auto ij = LocalTileIndex{i, j};
        // C = C - V W2
        //hpx::dataflow(executor_np, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans, NoTrans, -1.0, mat_vv.read(ik), mat_w2.read(kj), 1.0, std::move(mat_c(ij)));
	hpx::dataflow(executor_np, matrix::unwrapExtendTiles(tile::gemm_o), NoTrans, NoTrans, T(-1.0), mat_vv.read(ik), mat_w2.read(kj), T(1.0), std::move(mat_c(ij)));
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
#define DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(KWORD, DATATYPE) \
  KWORD template struct BackTransformation<Backend::MC, Device::CPU, DATATYPE>;

DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(extern, float)
DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(extern, double)
DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(extern, std::complex<float>)
DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(extern, std::complex<double>)

}
}
}
