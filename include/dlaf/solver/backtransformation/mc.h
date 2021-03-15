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

#include "dlaf/solver/backtransformation/api.h"

#include "dlaf/blas_tile.h"
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

#include "dlaf/matrix/matrix_output.h"

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
                      Matrix<const T, Device::CPU>& mat_v, common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus);
};

template <class T>
void BackTransformation<Backend::MC, Device::CPU, T>::call_FC(Matrix<T, Device::CPU>& mat_c,
                                                              Matrix<const T, Device::CPU>& mat_v,
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

  parallel_executor executor_hp(&get_thread_pool("default"), thread_priority::high);
  parallel_executor executor_normal(&get_thread_pool("default"), thread_priority::default_);

  const SizeType m = mat_c.nrTiles().rows();
  const SizeType n = mat_c.nrTiles().cols();
  const SizeType mb = mat_c.blockSize().rows();
  const SizeType nb = mat_c.blockSize().cols();

  // Matrix T
  comm::CommunicatorGrid comm_grid(MPI_COMM_WORLD, 1, 1, common::Ordering::ColumnMajor);
  common::Pipeline<comm::CommunicatorGrid> serial_comm(comm_grid);
  int tottaus;
  if (m < mb || m == 0 || n == 0)
    tottaus = 0;
  else
    tottaus = (m / mb - 1) * mb + m % mb;
  
  LocalElementSize sizeT(tottaus, tottaus);
  TileElementSize blockSizeT(mb, mb);
  Matrix<T, Device::CPU> mat_t(sizeT, blockSizeT);
  set_zero(mat_t);
  std::cout << "matrix T " << mat_t << std::endl; 
  std::cout << "tottaus " << tottaus << std::endl;
  
  Matrix<T, Device::CPU> mat_vv({mat_v.size().rows(), mb}, mat_v.blockSize());
  Matrix<T, Device::CPU> mat_w({mat_v.size().rows(), mb}, mat_v.blockSize());
  Matrix<T, Device::CPU> mat_w2({mb, mat_c.size().cols()}, mat_c.blockSize());

  SizeType last_mb;

  if (mat_v.blockSize().cols() == 1) {
    last_mb = 1;
  }
  else {
    if (mat_v.size().cols() % mat_v.blockSize().cols() == 0)
      last_mb = mat_v.blockSize().cols();
    else
      last_mb = mat_v.size().cols() % mat_v.blockSize().cols();
  }
  
  Matrix<T, Device::CPU> mat_vv_last({mat_v.size().rows(), last_mb}, mat_v.blockSize());
  Matrix<T, Device::CPU> mat_w_last({mat_v.size().rows(), last_mb}, mat_v.blockSize());
  Matrix<T, Device::CPU> mat_w2_last({last_mb, mat_c.size().cols()}, mat_c.blockSize());

  const SizeType reflectors = mat_v.size().cols() / mat_v.blockSize().cols() - 2;

  std::cout << "reflectors " << reflectors << std::endl;
  
  for (SizeType k = reflectors; k > -1; --k) {
    bool is_last = (k == reflectors) ? true : false;

    void (&cpyReg)(TileElementSize, TileElementIndex, const matrix::Tile<const T, Device::CPU>&,
                   TileElementIndex, const matrix::Tile<T, Device::CPU>&) = copy<T>;
    void (&cpy)(const matrix::Tile<const T, Device::CPU>&, const matrix::Tile<T, Device::CPU>&) =
        copy<T>;

    // Copy V panel into VV
    for (SizeType i = 0; i < mat_v.nrTiles().rows(); ++i) {
      if (is_last) {
        TileElementSize region = mat_vv_last.read(LocalTileIndex(i, 0)).get().size();
        TileElementIndex idx_in(0, 0);
        TileElementIndex idx_out(0, 0);
        hpx::dataflow(executor_hp, hpx::util::unwrapping(cpyReg), region, idx_in,
                      mat_v.read(LocalTileIndex(i, k)), idx_out, mat_vv_last(LocalTileIndex(i, 0)));
      }
      else {
        hpx::dataflow(executor_hp, hpx::util::unwrapping(cpy), mat_v.read(LocalTileIndex(i, k)),
                      mat_vv(LocalTileIndex(i, 0)));
      }

      // Fixing elements of VV and copying them into WH
      if (is_last) {
        auto tile_v = mat_vv_last(LocalTileIndex{i, 0}).get();
        if (i <= k) {
          lapack::laset(lapack::MatrixType::General, tile_v.size().rows(), tile_v.size().cols(), 0, 0,
                        tile_v.ptr(), tile_v.ld());
        }
        else if (i == k + 1) {
          lapack::laset(lapack::MatrixType::Upper, tile_v.size().rows(), tile_v.size().cols(), 0, 1,
                        tile_v.ptr(), tile_v.ld());
        }
        hpx::dataflow(executor_hp, hpx::util::unwrapping(cpy), mat_vv_last.read(LocalTileIndex(i, 0)),
                      mat_w_last(LocalTileIndex(i, 0)));
      }
      else {
        auto tile_v = mat_vv(LocalTileIndex{i, 0}).get();
        if (i <= k) {
          lapack::laset(lapack::MatrixType::General, tile_v.size().rows(), tile_v.size().cols(), 0, 0,
                        tile_v.ptr(), tile_v.ld());
        }
        else if (i == k + 1) {
          lapack::laset(lapack::MatrixType::Upper, tile_v.size().rows(), tile_v.size().cols(), 0, 1,
                        tile_v.ptr(), tile_v.ld());
        }
        hpx::dataflow(executor_hp, hpx::util::unwrapping(cpy), mat_vv.read(LocalTileIndex(i, 0)),
                      mat_w(LocalTileIndex(i, 0)));
      }
    }

    int taupan;
    // Reset W2 to zero
    if (is_last) {
      matrix::util::set(mat_w2_last, [](auto&&) { return 0; });
      taupan = last_mb;
    }
    else {
      matrix::util::set(mat_w2, [](auto&&) { return 0; });
      taupan = mat_v.blockSize().cols();
    }
    
    // Matrix T
  comm::CommunicatorGrid comm_grid(MPI_COMM_WORLD, 1, 1, common::Ordering::ColumnMajor);
  common::Pipeline<comm::CommunicatorGrid> serial_comm(comm_grid);

  const GlobalTileIndex v_start{0, k};
  std::cout << " v start " << v_start << " taupan " << taupan << std::endl;
  std::cout << "mat_v " << mat_v << std::endl;
  
  dlaf::factorization::internal::computeTFactor<Backend::MC>(taupan, mat_v, v_start, taus[k], mat_t(LocalTileIndex{k,k}), serial_comm);
  
    for (SizeType i = k + 1; i < m; ++i) {
      auto kk = LocalTileIndex{k, k};
      // WH = V T
      auto ik = LocalTileIndex{i, 0};
      if (is_last) {
	std::cout << "trmm last" << std::endl;
        hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::trmm<T, Device::CPU>), Right, Upper,
                      ConjTrans, NonUnit, 1.0, mat_t.read(kk), std::move(mat_w_last(ik)));
      }
      else {
	std::cout << "trmm" << std::endl;
        hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::trmm<T, Device::CPU>), Right, Upper,
                      ConjTrans, NonUnit, 1.0, mat_t.read(kk), std::move(mat_w(ik)));
      }
    }

    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{0, j};
      for (SizeType i = k + 1; i < m; ++i) {
        auto ik = LocalTileIndex{i, 0};
        auto ij = LocalTileIndex{i, j};
        // W2 = W C
        if (is_last) {
	  std::cout << "gemm1 last" << std::endl;
          hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans,
                        NoTrans, 1.0, std::move(mat_w_last(ik)), mat_c.read(ij), 1.0,
                        std::move(mat_w2_last(kj)));
        }
        else {
	  std::cout << "gemm1" << std::endl;
          hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans,
                        NoTrans, 1.0, std::move(mat_w(ik)), mat_c.read(ij), 1.0, std::move(mat_w2(kj)));
        }
      }
    }

    for (SizeType i = k + 1; i < m; ++i) {
      auto ik = LocalTileIndex{i, 0};
      for (SizeType j = 0; j < n; ++j) {
        auto kj = LocalTileIndex{0, j};
        auto ij = LocalTileIndex{i, j};
        // C = C - V W2
        if (is_last) {
	  std::cout << "gemm2 last" << std::endl;

          hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
                        NoTrans, -1.0, mat_vv_last.read(ik), mat_w2_last.read(kj), 1.0,
                        std::move(mat_c(ij)));
        }
        else {
	  std::cout << "gemm2" << std::endl;
          hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
                        NoTrans, -1.0, mat_vv.read(ik), mat_w2.read(kj), 1.0, std::move(mat_c(ij)));
        }
      }
    }

    printElements(mat_c);
  }
}

template <class T>
void BackTransformation<Backend::MC, Device::CPU, T>::call_FC(comm::CommunicatorGrid grid,
                                                              Matrix<T, Device::CPU>& mat_c,
                                                              Matrix<const T, Device::CPU>& mat_v,
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
