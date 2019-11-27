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

#include "dlaf/blas_tile.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix.h"
#include "dlaf/util_matrix.h"

/// @file

namespace dlaf {

/// @brief Cholesky implementation on local memory
///
/// @param uplo specifies that the matrix is \a Lower triangular
/// @tparam mat refers to a dlaf::Matrix object
///
/// @throws std::runtime_error if \p uplo = \a Upper decomposition is chosen (not yet implemented)

template <class T>
void cholesky_local(blas::Uplo uplo, Matrix<T, Device::CPU>& mat) {
  // Set up executor on the default queue with high priority.
  hpx::threads::scheduled_executor matrix_HP_executor =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_high);

  // Set up executor on the default queue with default priority.
  hpx::threads::scheduled_executor matrix_normal_executor =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_default);

  // Check if matrix is square
  util_matrix::assert_size_square(mat, "Cholesky", "mat");
  // Check if block matrix is square
  util_matrix::assert_blocksize_square(mat, "Cholesky", "mat");

  // Number of tile (rows = cols)
  SizeType nrtile = mat.nrTiles().cols();

  if (uplo == blas::Uplo::Lower) {
    for (SizeType k = 0; k < nrtile; ++k) {
      // Cholesky decomposition on mat(k,k) r/w potrf (lapack operation)
      auto kk = LocalTileIndex{k, k};

      hpx::dataflow(matrix_HP_executor, hpx::util::unwrapping(tile::potrf<T, Device::CPU>), uplo,
                    std::move(mat(kk)));

      for (SizeType i = k + 1; i < nrtile; ++i) {
        // Update panel mat(i,k) with trsm (blas operation), using data mat.read(k,k)
        hpx::dataflow(matrix_HP_executor, hpx::util::unwrapping(tile::trsm<T, Device::CPU>),
                      blas::Side::Right, uplo, blas::Op::ConjTrans, blas::Diag::NonUnit, 1.0,
                      mat.read(kk), std::move(mat(LocalTileIndex{i, k})));
      }

      for (SizeType j = k + 1; j < nrtile; ++j) {
        // Choose queue priority
        auto trailing_matrix_executor = (j == k + 1) ? matrix_HP_executor : matrix_normal_executor;

        // Update trailing matrix: diagonal element mat(j,j, reading mat.read(j,k), using herk (blas operation)
        hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::herk<T, Device::CPU>), uplo,
                      blas::Op::NoTrans, -1.0, mat.read(LocalTileIndex{j, k}), 1.0,
                      std::move(mat(LocalTileIndex{j, j})));

        for (SizeType i = j + 1; i < nrtile; ++i) {
          // Update remaining trailing matrix mat(i,j), reading mat.read(i,k) and mat.read(j,k), using
          // gemm (blas operation)
          hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>),
                        blas::Op::NoTrans, blas::Op::ConjTrans, -1.0, mat.read(LocalTileIndex{i, k}),
                        mat.read(LocalTileIndex{j, k}), 1.0, std::move(mat(LocalTileIndex{i, j})));
        }
      }
    }
  }
  else {
    throw std::runtime_error("uplo = Upper not yet implemented");
  }
}
}
