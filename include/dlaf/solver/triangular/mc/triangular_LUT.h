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
#include <hpx/runtime/threads/executors/pool_executor.hpp>

#include "dlaf/blas_tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace internal {
namespace mc {

// Local implementation of Left Upper Trans/ConjTrans
template <class T>
void triangular_LUT(blas::Op op, blas::Diag diag, T alpha, Matrix<const T, Device::CPU>& mat_a,
                    Matrix<T, Device::CPU>& mat_b) {
  constexpr auto Left = blas::Side::Left;
  constexpr auto Upper = blas::Uplo::Upper;
  constexpr auto NoTrans = blas::Op::NoTrans;

  using std::move;

  using hpx::threads::executors::pool_executor;
  using hpx::threads::thread_priority_high;
  using hpx::threads::thread_priority_default;
  using hpx::util::unwrapping;
  using hpx::dataflow;

  // Set up executor on the default queue with high priority.
  pool_executor executor_hp("default", thread_priority_high);
  // Set up executor on the default queue with default priority.
  pool_executor executor_normal("default", thread_priority_default);

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < m; ++k) {
    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{k, j};

      // Triangular solve of k-th row Panel of B
      auto trsm_f = unwrapping([Left, Upper, op, diag, alpha](auto&& kk_tile, auto&& kj_tile) {
        tile::trsm<T, Device::CPU>(Left, Upper, op, diag, alpha, kk_tile, kj_tile);
      });
      dataflow(executor_hp, move(trsm_f), mat_a.read(LocalTileIndex{k, k}), move(mat_b(kj)));

      for (SizeType i = k + 1; i < m; ++i) {
        // Choose queue priority
        auto trailing_executor = (i == k + 1) ? executor_hp : executor_normal;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        auto gemm_f = unwrapping([op, NoTrans, beta](auto&& ki_tile, auto&& kj_tile, auto&& ij_tile) {
          tile::gemm<T, Device::CPU>(op, NoTrans, beta, ki_tile, kj_tile, 1.0, ij_tile);
        });
        dataflow(trailing_executor, move(gemm_f), mat_a.read(LocalTileIndex{k, i}), mat_b.read(kj),
                 move(mat_b(LocalTileIndex{i, j})));
      }
    }
  }
}

}
}
}
