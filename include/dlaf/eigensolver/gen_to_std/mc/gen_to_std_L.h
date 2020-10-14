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
#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/tile_output.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace internal {
namespace mc {

// Implementation based on LAPACK Algorithm for the transformation from generalized to standard
// eigenproblem (xHEGST)

// Local implementation
template <class T>
void genToStd_L(Matrix<T, Device::CPU>& mat_a, Matrix<T, Device::CPU>& mat_l) {
  constexpr auto Right = blas::Side::Right;
  constexpr auto Left = blas::Side::Left;
  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto NonUnit = blas::Diag::NonUnit;
  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto ConjTrans = blas::Op::ConjTrans;

  using hpx::util::unwrapping;

  using hpx::threads::executors::pool_executor;
  using hpx::threads::thread_priority_high;
  using hpx::threads::thread_priority_default;

  // Set up executor on the default queue with high priority.
  pool_executor executor_hp("default", thread_priority_high);

  // Set up executor on the default queue with default priority.
  pool_executor executor_normal("default", thread_priority_default);

  SizeType m = mat_a.nrTiles().rows();
  SizeType n = mat_a.nrTiles().cols();

  for (SizeType k = 0; k < n; ++k) {
    auto kk = LocalTileIndex{k, k};

    // Direct transformation to standard eigenvalue problem of the diagonal tile
    hpx::dataflow(executor_hp, unwrapping(tile::hegst<T, Device::CPU>), 1, Lower, std::move(mat_a(kk)),
                  std::move(mat_l(kk)));

    if (k != (n - 1)) {
      for (SizeType i = k + 1; i < m; ++i) {
        // Working on panel...
        auto ik = LocalTileIndex{i, k};
        hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), Right, Lower,
                      ConjTrans, NonUnit, 1.0, mat_l.read(kk), std::move(mat_a(ik)));
        hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::hemm<T, Device::CPU>), Right, Lower,
                      -0.5, mat_a.read(kk), mat_l.read(ik), 1.0, std::move(mat_a(ik)));
      }

      for (SizeType j = k + 1; j < n; ++j) {
        // Working on trailing matrix...
        auto jj = LocalTileIndex{j, j};
        auto jk = LocalTileIndex{j, k};
        hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::her2k<T, Device::CPU>), Lower, NoTrans,
                      -1.0, mat_a.read(jk), mat_l.read(jk), 1.0, std::move(mat_a(jj)));

        for (SizeType i = j + 1; i < m; ++i) {
          auto ik = LocalTileIndex{i, k};
          auto ij = LocalTileIndex{i, j};
          hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
                        ConjTrans, -1.0, mat_a.read(ik), mat_l.read(jk), 1.0, std::move(mat_a(ij)));
          hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
                        ConjTrans, -1.0, mat_l.read(ik), mat_a.read(jk), 1.0, std::move(mat_a(ij)));
        }
      }

      for (SizeType i = k + 1; i < m; ++i) {
        // Working on panel...
        auto ik = LocalTileIndex{i, k};
        hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::hemm<T, Device::CPU>), Right, Lower, -0.5,
                      mat_a.read(kk), mat_l.read(ik), 1.0, std::move(mat_a(ik)));
      }

      for (SizeType j = k + 1; j < n; ++j) {
        auto jj = LocalTileIndex{j, j};
        auto jk = LocalTileIndex{j, k};
        hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), Left, Lower,
                      NoTrans, NonUnit, 1.0, mat_l.read(jj), std::move(mat_a(jk)));

        for (SizeType i = j + 1; i < m; ++i) {
          auto ij = LocalTileIndex{i, j};
          auto ik = LocalTileIndex{i, k};
          hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
                        NoTrans, -1.0, mat_l.read(ij), mat_a.read(jk), 1.0, std::move(mat_a(ik)));
        }
      }
    }
  }
}

}
}
}
