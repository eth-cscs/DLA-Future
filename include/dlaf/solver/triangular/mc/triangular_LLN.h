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
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace internal {
namespace mc {

// Local implementation of Left Lower NoTrans
template <class T>
void triangular_LLN(blas::Diag diag, T alpha, Matrix<const T, Device::CPU>& mat_a,
                    Matrix<T, Device::CPU>& mat_b) {
  constexpr auto Left = blas::Side::Left;
  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto NoTrans = blas::Op::NoTrans;

  // Set up executor on the default queue with high priority.
  hpx::threads::scheduled_executor executor_hp =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_high);

  // Set up executor on the default queue with default priority.
  hpx::threads::scheduled_executor executor_normal =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_default);

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < m; ++k) {
    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{k, j};

      // Triangular solve of the first tile
      hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), Left, Lower, NoTrans,
                    diag, alpha, mat_a.read(LocalTileIndex{k, k}), std::move(mat_b(kj)));

      for (SizeType i = k + 1; i < m; ++i) {
        // Choose queue priority
        auto trailing_executor = (i == k + 1) ? executor_hp : executor_normal;
        auto beta = static_cast<T>(-1.0) / alpha;
        // Matrix multiplication
        hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
                      blas::Op::NoTrans, beta, mat_a.read(LocalTileIndex{i, k}), mat_b.read(kj), 1.0,
                      std::move(mat_b(LocalTileIndex{i, j})));
      }
    }
  }
}

// Distributed implementation of Left Lower NoTrans
template <class T>
void triangular_LLN(comm::CommunicatorGrid grid, blas::Diag diag, T alpha,
                    Matrix<const T, Device::CPU>& mat_a, Matrix<T, Device::CPU>& mat_b) {
  constexpr auto Left = blas::Side::Left;
  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto NoTrans = blas::Op::NoTrans;

  // Set up executor on the default queue with high priority.
  hpx::threads::scheduled_executor executor_hp =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_high);

  // Set up executor on the default queue with default priority.
  hpx::threads::scheduled_executor executor_normal =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_default);

  hpx::threads::scheduled_executor executor_mpi;
  try {
    executor_mpi = hpx::threads::executors::pool_executor("mpi", hpx::threads::thread_priority_high);
  }
  catch (...) {
    executor_mpi = executor_hp;
  }
}
}
}
}
