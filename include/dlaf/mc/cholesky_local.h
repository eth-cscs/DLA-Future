// Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include "dlaf/blas_tile.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/util_blas.h"
#include "dlaf/util_matrix.h"
//
#include <exception>
//
#include <hpx/runtime/threads/run_as_hpx_thread.hpp>
#include "hpx/hpx.hpp"
#include "hpx/include/parallel_executors.hpp"
#include "hpx/include/threads.hpp"

/// @file

namespace dlaf {

/// @brief Cholesky implementation on local memory
///
/// @tparam mat referes to a dlaf::Matrix object
/// @param uplo specifies whether the matrix is \a Upper or \a Lower triangular
/// @param side defines whether the matrix is on the \a Left or \a Right side of the unknown matrix
/// @param op specifies the form of \p mat used in matrix multiplication: \a NoTrans, \a Trans, \a ConjTrans
/// @param diag describes if the matrix is unit triangular (\a Unit) or not (\a NonUnit)
///
/// @throws std::runtime_error if \p uplo = \a Upper decomposition  is chosen (not yet implemented)

template <class T>
void cholesky_local(Matrix<T, Device::CPU>& mat, blas::Uplo uplo, blas::Side side, blas::Op op,
                    blas::Diag diag) {
  /// First, two executors are set up on the default queue: with high and with default priority.

  // Set up executor on the default queue with high priority.
  hpx::threads::scheduled_executor matrix_HP_executor =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_high);

  // Set up executor on the default queue with default priority.
  hpx::threads::scheduled_executor matrix_normal_executor =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_default);

  /// Then the matrix and its sublocks are tested to confirm they are square.
  // Check if matrix is square
  util_matrix::check_size_square(mat, "Cholesky", "mat");
  // Check if block matrix is square
  util_matrix::check_blocksize_square(mat, "Cholesky", "mat");

  // Number of tile (rows = cols)
  SizeType nrtile = mat.nrTiles().cols();

  // k-loop
  for (SizeType k = 0; k < nrtile; ++k) {
    /// A Cholesky decomposition with the tile::potrf function is done on the diagonal tiles.
    // Cholesky decomposition on mat(k,k) r/w potrf (lapack operation)
    if (uplo == blas::Uplo::Lower) {
      hpx::dataflow(hpx::util::unwrapping(tile::potrf<T, Device::CPU>), uplo, std::move(mat({k, k})));
    }
    else {
      throw std::runtime_error("uplo = Upper not yet implemented");
    }

    // i-loop
    for (SizeType i = k + 1; i < nrtile; ++i) {
      /// The panel below each diagonal tile is updated, using tile::trsm.
      // Update panel mat(i,k) with trsm (blas operation), using data mat.read(k,k)
      if (uplo == blas::Uplo::Lower && side == blas::Side::Right && op == blas::Op::ConjTrans &&
          diag == blas::Diag::NonUnit) {
        T alpha = 1.0;
        hpx::dataflow(hpx::util::unwrapping(tile::trsm<T, Device::CPU>), side, uplo, op, diag, alpha,
                      mat.read({k, k}), std::move(mat({i, k})));
      }
      else {
        throw std::runtime_error(
            "uplo = Upper, side = Left, op = Trans/NoTrans, diag = Unit not yet implemented");
      }
    }

    // j-loop
    for (SizeType j = k + 1; j < nrtile; ++j) {
      /// Next, also the diagonal tiles of the trailing matrix are updated with tile::herk operation.
      // Update trailing matrix: diagonal element mat(j,j, reading mat.read(j,k), using herk (blas operation)
      if (uplo == blas::Uplo::Lower && op == blas::Op::ConjTrans) {
        BaseType<T> alpha = -1.0;
        BaseType<T> beta = 1.0;
        hpx::dataflow(hpx::util::unwrapping(tile::herk<T, Device::CPU>), uplo, blas::Op::NoTrans, alpha,
                      mat.read({j, k}), beta, std::move(mat({j, j})));
      }
      else {
        throw std::runtime_error("uplo = Upper, diag = Unit not yet implemented");
      }

      // internal i-loop
      for (SizeType i = j + 1; i < nrtile; ++i) {
        /// Finally, the remaining trailing matrix is updated with tile::gemm.
        // Update remaining trailing matrix mat(i,j), reading mat.read(i,k) and mat.read(j,k), using gemm
        // (blas operation)
        T alpha = -1.0;
        T beta = 1.0;
        hpx::dataflow(hpx::util::unwrapping(tile::gemm<T, Device::CPU>), blas::Op::NoTrans,
                      blas::Op::ConjTrans, alpha, mat.read({i, k}), mat.read({j, k}), beta,
                      std::move(mat({i, j})));
      }
    }
  }
}
}
