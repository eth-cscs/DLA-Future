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
#include <hpx/include/util.hpp>
#include <hpx/local/future.hpp>

#include <blas.hh>

#include "dlaf/solver/backtransformation/api.h"

#include "dlaf/blas_tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/vector.h"
//#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix/copy.h"
//#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/util_matrix.h"

#include "dlaf/matrix_output.h"

namespace dlaf {
namespace solver {
namespace internal {

  using namespace dlaf::matrix;
  //  using namespace dlaf::tile;
  
// Implementation based on:
// 1. Part of algorithm 6 "LAPACK Algorithm for the eigenvector back-transformation", page 15, PhD thesis "GPU
// Accelerated Implementations of a Generalized Eigenvalue Solver for Hermitian Matrices with Systematic
// Energy and Time to Solution Analysis" presented by Raffaele Solcà (2016)
// 2. Report "Gep + back-transformation", Alberto Invernizzi (2020)
// 3. Report "Reduction to band + back-transformation", Raffaele Solcà (2020)

template <class T>
struct BackTransformation<Backend::MC, Device::CPU, T> {
  static void call_FC(Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v,
		      Matrix<T, Device::CPU>& mat_t);
 };

 
 template <class T>
   void BackTransformation<Backend::MC, Device::CPU, T>::call_FC(Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v, Matrix<T, Device::CPU>& mat_t)
   {
     constexpr auto Left = blas::Side::Left;
     constexpr auto Right = blas::Side::Right;
     constexpr auto Upper = blas::Uplo::Upper;
     constexpr auto Lower = blas::Uplo::Lower;
     constexpr auto NoTrans = blas::Op::NoTrans;
     constexpr auto ConjTrans = blas::Op::ConjTrans;
     constexpr auto NonUnit = blas::Diag::NonUnit;

     using hpx::threads::executors::pool_executor;
     using hpx::threads::thread_priority_high;
     using hpx::threads::thread_priority_default;
     using hpx::util::unwrapping;
     
     // Set up executor on the default queue with high priority.
     pool_executor executor_hp("default", thread_priority_high);
     // Set up executor on the default queue with default priority.
     pool_executor executor_normal("default", thread_priority_default);

     SizeType m = mat_c.nrTiles().rows();
     SizeType n = mat_c.nrTiles().cols();
     SizeType mb = mat_c.blockSize().rows();
     SizeType nb = mat_c.blockSize().cols();

     TileElementSize size(mb, nb);

     // CHECK!!
     auto dist_w = mat_c.distribution();
     auto layout_w = tileLayout(dist_w.localSize(), size);
     Matrix<T, Device::CPU> mat_w(std::move(dist_w), layout_w);
     copy(mat_v, mat_w);

     auto dist_w2 = mat_c.distribution();
     auto layout_w2 = tileLayout(dist_w2.localSize(), size);
     Matrix<T, Device::CPU> mat_w2(std::move(dist_w2), layout_w2);
     matrix::util::set(mat_w2, [](auto&&){return 0;});

     // n-1 reflectors
     for (SizeType i = 0; i < (m - 1); ++i) {

       for (SizeType k = i; k < m; ++k) {
	 auto ki = LocalTileIndex{k, i};
	 auto ii = LocalTileIndex{i, i};
	 // W = V T
	 hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trmm<T, Device::CPU>), Right, Upper, ConjTrans,
		       NonUnit, 1.0, mat_t.read(ii), std::move(mat_w(ki)));
       }

       for (SizeType k = i; k < m; ++k) {
	 auto ik = LocalTileIndex{i, k};
	 for (SizeType j = i; j < m; ++j) {
	   auto ji = LocalTileIndex{j, i};
	   auto jk = LocalTileIndex{j, k};
	   // W2 = WH C
	   hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans,
			 NoTrans, 1.0, std::move(mat_w(ji)), mat_c.read(jk), 1.0, std::move(mat_w2(ik)));
	 }
       }

       for (SizeType k = i; k < m; ++k) {
	 auto ki = LocalTileIndex{k, i};
	 for (SizeType j = i; j < m; ++j) {
	   auto ij = LocalTileIndex{i, j};
	   auto kj = LocalTileIndex{k, j};
	   // C = C - V W2
	   hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
			 NoTrans, -1.0, mat_v.read(ki), mat_w2.read(ij), 1.0, std::move(mat_c(kj)));
	 }
       }
     }
   }

 
/// ---- ETI
#define DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(KWORD, DATATYPE)		\
  KWORD template struct BackTransformation<Backend::MC, Device::CPU, DATATYPE>;

 DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(extern, float)
 DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(extern, double)
 DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(extern, std::complex<float>)
 DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(extern, std::complex<double>)

}
}
}



