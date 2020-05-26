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
  constexpr auto Trans = blas::Op::Trans;

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


    // BELOW: TO BE MODIFIED!!!
    //        if (k != (n - 1)) {
//     for (SizeType j = k + 1; j < m; ++j) {
//       // Working on panel...
//        auto jk = LocalTileIndex{j, k};
//
//        // TRSM
//       hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), Right, Lower,
//                      ConjTrans, NonUnit, 1.0, mat_l.read(kk), std::move(mat_a(jk)));
//        // HEMM
//        hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::hemm<T, Device::CPU>), Right, Lower, -0.5,
//                      mat_a.read(kk), mat_l.read(jk), 1.0, std::move(mat_a(jk)));
//      }
//
//      for (SizeType j = k + 1; j < m; ++j) {
//        for (SizeType i = k + 1; j < n; ++i) {
//          // Working on trailing matrix...
//          auto jk = LocalTileIndex{j, k};
//          auto ik = LocalTileIndex{i, k};
//          auto ji = LocalTileIndex{j, i};
//
//          // GEMM
//          hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
//                        ConjTrans, -1.0, mat_a.read(jk), mat_l.read(ik), 1.0, std::move(mat_a(ji)));
//          // GEMM
//          hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
//                        ConjTrans, -1.0, mat_l.read(jk), mat_a.read(ik), 1.0, std::move(mat_a(ji)));
//        }
//      }
//
//      for (SizeType j = k + 1; j < m; ++j) {
//        // Working on panel...
//        auto jk = LocalTileIndex{j, k};
//
//        // HEMM
//        hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::hemm<T, Device::CPU>), Right, Lower, -0.5,
//                      mat_a.read(kk), mat_l.read(jk), 1.0, std::move(mat_a(jk)));
//      }
//
//      //memory::MemoryView<T, Device::CPU> memory_view(m * n);
//      
//      //TileElementSize size = mat_a.blockSize();
//      //auto mem_view = memory_view;  // Copy the memory view to check the elements later.
//
//      for (SizeType j = k + 1; j < m; ++j) {
//	//Tile<T, Device::CPU> temp_tile(size, std::move(mem_view), m);
//
//        for (SizeType i = k + 1; j < n; ++i) {
//          // Working on trailing matrix...
//          auto ij = LocalTileIndex{i, j};
//          auto ki = LocalTileIndex{k, i};
//          auto tile = mat_a(ij);
//
//          // TRSM
//          hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), Left, Lower,
//                        NoTrans, NonUnit, 1.0, mat_l.read(ki), std::move(tile));
//
//          //not implemented yet
//	  //temp_tile += tile;
//	  //temp_tile = temp_tile + std::move(tile);
//
//	  //std::cout << "MAO " << temp_tile.size() << std::endl;
//        }
//        //mat_a(LocalTileIndex{j, k}).get() = temp_tile;
//      }
//    }
  }
}

}
}
}
