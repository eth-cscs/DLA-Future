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

// Local implementation of Right Lower NoTrans
template <class T>
void triangular_RLN(blas::Diag diag, T alpha, Matrix<const T, Device::CPU>& mat_a,
                    Matrix<T, Device::CPU>& mat_b) {
  constexpr auto Right = blas::Side::Right;
  constexpr auto Lower = blas::Uplo::Lower;
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

  for (SizeType k = n - 1; k > -1; --k) {
    for (SizeType i = m - 1; i > -1; --i) {
      auto ik = LocalTileIndex{i, k};

      // Triangular solve of k-th col Panel of B
      auto trsm_f = unwrapping([Right, Lower, NoTrans, diag, alpha](auto&& a_tile, auto&& b_tile) {
        tile::trsm<T, Device::CPU>(Right, Lower, NoTrans, diag, alpha, a_tile, b_tile);
      });
      dataflow(executor_hp, move(trsm_f), mat_a.read(LocalTileIndex{k, k}), move(mat_b(ik)));

      for (SizeType j = k - 1; j > -1; --j) {
        // Choose queue priority
        auto trailing_executor = (j == k - 1) ? executor_hp : executor_normal;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        auto gemm_f = unwrapping([NoTrans, beta](auto&& ik_tile, auto&& a_tile, auto&& ij_tile) {
          tile::gemm<T, Device::CPU>(NoTrans, NoTrans, beta, ik_tile, a_tile, 1.0, ij_tile);
        });
        dataflow(trailing_executor, move(gemm_f), mat_b.read(ik), mat_a.read(LocalTileIndex{k, j}),
                 move(mat_b(LocalTileIndex{i, j})));
      }
    }
  }
}

}
}
}
