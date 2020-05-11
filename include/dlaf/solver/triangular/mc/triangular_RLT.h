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

// Local implementation of Right Lower Trans/ConjTrans
template <class T>
void triangular_RLT(blas::Op op, blas::Diag diag, T alpha, Matrix<const T, Device::CPU>& mat_a,
                    Matrix<T, Device::CPU>& mat_b) {
  constexpr auto Right = blas::Side::Right;
  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto NoTrans = blas::Op::NoTrans;

  using hpx::threads::executors::pool_executor;
  using hpx::threads::thread_priority_high;
  using hpx::threads::thread_priority_default;

  // Set up executor on the default queue with high priority.
  pool_executor executor_hp("default", thread_priority_high);
  // Set up executor on the default queue with default priority.
  pool_executor executor_normal("default", thread_priority_default);

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < n; ++k) {
    for (SizeType i = 0; i < m; ++i) {
      auto ik = LocalTileIndex{i, k};

      // Triangular solve of k-th col Panel of B
      hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), Right, Lower, op,
                    diag, alpha, mat_a.read(LocalTileIndex{k, k}), std::move(mat_b(ik)));

      for (SizeType j = k + 1; j < n; ++j) {
        // Choose queue priority
        auto trailing_executor = (j == k - 1) ? executor_hp : executor_normal;

        auto beta = static_cast<T>(-1.0) / alpha;
        // Update trailing matrix
        hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans, op,
                      beta, mat_b.read(ik), mat_a.read(LocalTileIndex{j, k}), 1.0,
                      std::move(mat_b(LocalTileIndex{i, j})));
      }
    }
  }
}

}
}
}
