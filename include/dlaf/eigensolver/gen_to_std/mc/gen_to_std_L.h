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
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace internal {
namespace mc {

// Local implementation
template <class T>
void genToStd(Matrix<T, Device::CPU>& mat_a, Matrix<const T, Device::CPU>& mat_l) {
  constexpr auto Right = blas::Side::Right;
  constexpr auto Left = blas::Side::Left;
  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto NonUnit = blas::Diag::NonUnit;
  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto ConjTrans = blas::Op::ConjTrans;

  // Set up executor on the default queue with high priority.
  hpx::threads::scheduled_executor executor_hp =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_high);

  // Set up executor on the default queue with default priority.
  hpx::threads::scheduled_executor executor_normal =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_default);
  SizeType m = mat_a.nrTiles().rows();
  SizeType n = mat_a.nrTiles().cols();

  // Choose queue priority
  // auto trailing_executor = (j == k - 1) ? executor_hp : executor_normal;

  // auto beta = static_cast<T>(-1.0) / alpha;
  // Update trailing matrix

  for (SizeType k = 0; k < n; ++k) {
    auto kk = LocalTileIndex{k, k};

    // Direct transformation to standard eigenvalue problem of the diagonal tile
    hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::hegst<T, Device::CPU>), 1, Lower, std::move(mat_a(kk)), mat_l.read(kk));

    if (k != (n - 1)) {
      for (SizeType i = k + 1; i < m; ++i) {
	// Working on panel...
        auto ik = LocalTileIndex{i, k};

        // TRSM
	hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), Right, Lower,
                      ConjTrans, NonUnit, 1.0, mat_l.read(kk), std::move(mat_a(ik)));
        // HEMM
        hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::hemm<T, Device::CPU>), Right, Lower, -0.5,
                      mat_a.read(kk), mat_l.read(ik), 1.0, std::move(mat_a(ik)));
      }

      for (SizeType j = k + 1; j < n; ++j) {
	// Working on trailing matrix...
        auto jj = LocalTileIndex{j, j};
        auto jk = LocalTileIndex{j, k};

        // HER2K
        hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::her2k<T, Device::CPU>), Lower, NoTrans, -1.0,
                      mat_a.read(jk), mat_l.read(jk), 1.0, std::move(mat_a(jj)));

	for (SizeType i = j + 1; j < m; ++i) {
          auto ik = LocalTileIndex{i, k};
          auto ij = LocalTileIndex{i, j};

          // GEMM
          hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
                        ConjTrans, -1.0, mat_a.read(ik), mat_l.read(jk), 1.0, std::move(mat_a(ij)));
          // GEMM
          hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
                        ConjTrans, -1.0, mat_l.read(ik), mat_a.read(jk), 1.0, std::move(mat_a(ij)));
        }
      }

      for (SizeType i = k + 1; i < m; ++i) {
        // Working on panel...
        auto ik = LocalTileIndex{i, k};

        // HEMM
        hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::hemm<T, Device::CPU>), Right, Lower, -0.5,
                      mat_a.read(kk), mat_l.read(ik), 1.0, std::move(mat_a(ik)));
      }

      for (SizeType j = k + 1; j < n; ++j) {
	auto jj = LocalTileIndex{j, j};
	auto jk = LocalTileIndex{j, k};
		
        // TRSM
	hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), Left, Lower,
                      NoTrans, NonUnit, 1.0, mat_l.read(jj), std::move(mat_a(jk)));

        for (SizeType i = j + 1; i < m; ++i) {
          // Working on trailing matrix...
          auto ij = LocalTileIndex{i, j};
          auto ik = LocalTileIndex{i, k};

          // GEMM
          hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
                        NoTrans, -1.0, mat_l.read(ij), mat_a.read(jk), 1.0, std::move(mat_a(ik)));

        }
      }
      
    }
  }
}

}
}
}
