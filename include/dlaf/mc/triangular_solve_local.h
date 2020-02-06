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
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/util_matrix.h"

/// @file

namespace dlaf {
/// @brief Triangular Solve implementation on local memory, solving op(A) X = alpha B (when side
/// == Left) or X op(A) = alpha B (when side == Right).
///
/// @param side specifies whether op(A) appears on the \a Left or on the \a Right of matrix X
/// @param uplo specifies whether the matrix A is a \a Lower or \a Upper triangular matrix
/// @param op specifies the form of op(A) to be used in the matrix multiplication: \a NoTrans, \a
/// Trans, \a ConjTrans
/// @param diag specifies if the matrix A is assumed to be unit triangular (\a Unit) or not (\a
/// NonUnit)
/// @param mat_a contains the triangular matrix A. Only the tiles of the matrix which contain the upper
/// or the lower triangular part (depending on the value of uplo) are accessed in read-only mode (the
/// elements are not modified).
/// @param mat_b on entry it contains the matrix B, on exit the matrix elements are overwritten with the
/// elements of the matrix X.

template <class T>
void triangular_solve(blas::Side side, blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha,
                      Matrix<const T, Device::CPU>& mat_a, Matrix<T, Device::CPU>& mat_b) {
  // Set up executor on the default queue with high priority.
  hpx::threads::scheduled_executor executor_hp =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_high);

  // Set up executor on the default queue with default priority.
  hpx::threads::scheduled_executor executor_normal =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_default);

  // Check if matrix A is square
  util_matrix::assertSizeSquare(mat_a, "TriangularSolve", "mat_a");
  // Check if block matrix A is square
  util_matrix::assertBlocksizeSquare(mat_a, "TriangularSolve", "mat_a");
  // Check if A and B dimensions are compatible
  util_matrix::assertMultipliableMatrices(mat_a, mat_b, side, op, "TriangularSolve", "mat_a", "mat_b");
  // Check if matrix A is stored on local memory
  util_matrix::assertLocalMatrix(mat_a, "TriangularSolve", "mat_a");
  // Check if matrix B is stored on local memory
  util_matrix::assertLocalMatrix(mat_b, "TriangularSolve", "mat_b");

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  if (uplo == blas::Uplo::Upper) {
    if (side == blas::Side::Left) {
      if (op == blas::Op::NoTrans) {
        // Upper Left NoTrans

        // Loop on rows of A matrix
        for (SizeType k = m - 1; k > -1; --k) {
          // Loop on cols of A matrix
          for (SizeType j = n - 1; j > -1; --j) {
            auto kj = LocalTileIndex{k, j};
            // Triangular solve of the first tile
            hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), side, uplo, op,
                          diag, alpha, mat_a.read(LocalTileIndex{k, k}), std::move(mat_b(kj)));

            for (SizeType i = k - 1; i > -1; --i) {
              // Choose queue priority
              auto trailing_executor = (i == k - 1) ? executor_hp : executor_normal;

              auto beta = static_cast<T>(-1.0) / alpha;
              // Matrix multiplication to update other eigenvectors
              hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), op,
                            blas::Op::NoTrans, beta, mat_a.read(LocalTileIndex{i, k}), mat_b.read(kj),
                            1.0, std::move(mat_b(LocalTileIndex{i, j})));
            }
          }
        }
      }
      else {
        // Upper Left Trans/ConjTrans case

        // Loop on rows of A matrix
        for (SizeType k = 0; k < m; ++k) {
          // Loop on cols of A matrix
          for (SizeType j = 0; j < n; ++j) {
            auto kj = LocalTileIndex{k, j};

            // Triangular solve of the first tile
            hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), side, uplo, op,
                          diag, alpha, mat_a.read(LocalTileIndex{k, k}), std::move(mat_b(kj)));

            for (SizeType i = k + 1; i < m; ++i) {
              // Choose queue priority
              auto trailing_executor = (i == k + 1) ? executor_hp : executor_normal;

              auto beta = static_cast<T>(-1.0) / alpha;
              // Matrix multiplication to update other eigenvectors
              hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), op,
                            blas::Op::NoTrans, beta, mat_a.read(LocalTileIndex{k, i}), mat_b.read(kj),
                            1.0, std::move(mat_b(LocalTileIndex{i, j})));
            }
          }
        }
      }
    }
    else {
      if (op == blas::Op::NoTrans) {
        // Upper Right NoTrans case

        // Loop on cols of A matrix
        for (SizeType k = 0; k < n; ++k) {
          // Loop on rows of A matrix
          for (SizeType i = 0; i < m; ++i) {
            auto ik = LocalTileIndex{i, k};

            // Triangular solve of the first tile
            hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), side, uplo, op,
                          diag, alpha, mat_a.read(LocalTileIndex{k, k}), std::move(mat_b(ik)));

            for (SizeType j = k + 1; j < n; ++j) {
              // Choose queue priority
              auto trailing_executor = (j == k - 1) ? executor_hp : executor_normal;

              auto beta = static_cast<T>(-1.0) / alpha;
              // Matrix multiplication to update other eigenvectors
              hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>),
                            blas::Op::NoTrans, op, beta, mat_b.read(ik),
                            mat_a.read(LocalTileIndex{k, j}), 1.0,
                            std::move(mat_b(LocalTileIndex{i, j})));
            }
          }
        }
      }
      else {
        // Upper Right Trans/ConjTrans case

        // Loop on cols of A matrix
        for (SizeType k = n - 1; k > -1; --k) {
          // Loop on rows of A matrix
          for (SizeType i = m - 1; i > -1; --i) {
            auto ik = LocalTileIndex{i, k};

            // Triangular solve of the first tile
            hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), side, uplo, op,
                          diag, alpha, mat_a.read(LocalTileIndex{k, k}), std::move(mat_b(ik)));

            for (SizeType j = k - 1; j > -1; --j) {
              // Choose queue priority
              auto trailing_executor = (j == k - 1) ? executor_hp : executor_normal;

              auto beta = static_cast<T>(-1.0) / alpha;
              // Matrix multiplication to update other eigenvectors
              hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>),
                            blas::Op::NoTrans, op, beta, mat_b.read(ik),
                            mat_a.read(LocalTileIndex{j, k}), 1.0,
                            std::move(mat_b(LocalTileIndex{i, j})));
            }
          }
        }
      }
    }
  }
  else {
    if (side == blas::Side::Left) {
      if (op == blas::Op::NoTrans) {
        // Lower Left NoTrans case

        // Loop on rows of A matrix
        for (SizeType k = 0; k < m; ++k) {
          // Loop on cols of A matrix
          for (SizeType j = 0; j < n; ++j) {
            auto kj = LocalTileIndex{k, j};

            // Triangular solve of the first tile
            hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), side, uplo, op,
                          diag, alpha, mat_a.read(LocalTileIndex{k, k}), std::move(mat_b(kj)));

            for (SizeType i = k + 1; i < m; ++i) {
              // Choose queue priority
              auto trailing_executor = (i == k + 1) ? executor_hp : executor_normal;

              auto beta = static_cast<T>(-1.0) / alpha;
              // Matrix multiplication to update other eigenvectors
              hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), op,
                            blas::Op::NoTrans, beta, mat_a.read(LocalTileIndex{i, k}), mat_b.read(kj),
                            1.0, std::move(mat_b(LocalTileIndex{i, j})));
            }
          }
        }
      }
      else {
        // Lower Left Trans/ConjTrans case

        // Loop on rows of A matrix
        for (SizeType k = m - 1; k > -1; --k) {
          // Loop on cols of A matrix
          for (SizeType j = n - 1; j > -1; --j) {
            auto kj = LocalTileIndex{k, j};
            // Triangular solve of the first tile
            hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), side, uplo, op,
                          diag, alpha, mat_a.read(LocalTileIndex{k, k}), std::move(mat_b(kj)));

            for (SizeType i = k - 1; i > -1; --i) {
              // Choose queue priority
              auto trailing_executor = (i == k - 1) ? executor_hp : executor_normal;

              auto beta = static_cast<T>(-1.0) / alpha;
              // Matrix multiplication to update other eigenvectors
              hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), op,
                            blas::Op::NoTrans, beta, mat_a.read(LocalTileIndex{k, i}), mat_b.read(kj),
                            1.0, std::move(mat_b(LocalTileIndex{i, j})));
            }
          }
        }
      }
    }
    else {
      if (op == blas::Op::NoTrans) {
        // Lower Right NoTrans case

        // Loop on cols of A matrix
        for (SizeType k = n - 1; k > -1; --k) {
          // Loop on rows of A matrix
          for (SizeType i = m - 1; i > -1; --i) {
            auto ik = LocalTileIndex{i, k};

            // Triangular solve of the first tile
            hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), side, uplo, op,
                          diag, alpha, mat_a.read(LocalTileIndex{k, k}), std::move(mat_b(ik)));

            for (SizeType j = k - 1; j > -1; --j) {
              // Choose queue priority
              auto trailing_executor = (j == k - 1) ? executor_hp : executor_normal;

              auto beta = static_cast<T>(-1.0) / alpha;
              // Matrix multiplication to update other eigenvectors
              hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>),
                            blas::Op::NoTrans, op, beta, mat_b.read(ik),
                            mat_a.read(LocalTileIndex{k, j}), 1.0,
                            std::move(mat_b(LocalTileIndex{i, j})));
            }
          }
        }
      }
      else {
        // Lower Right Trans/ConjTrans case

        // Loop on cols of A matrix
        for (SizeType k = 0; k < n; ++k) {
          // Loop on rows of A matrix
          for (SizeType i = 0; i < m; ++i) {
            auto ik = LocalTileIndex{i, k};

            // Triangular solve of the first tile
            hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), side, uplo, op,
                          diag, alpha, mat_a.read(LocalTileIndex{k, k}), std::move(mat_b(ik)));

            for (SizeType j = k + 1; j < n; ++j) {
              // Choose queue priority
              auto trailing_executor = (j == k - 1) ? executor_hp : executor_normal;

              auto beta = static_cast<T>(-1.0) / alpha;
              // Matrix multiplication to update other eigenvectors
              hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>),
                            blas::Op::NoTrans, op, beta, mat_b.read(ik),
                            mat_a.read(LocalTileIndex{j, k}), 1.0,
                            std::move(mat_b(LocalTileIndex{i, j})));
            }
          }
        }
      }
    }
  }
}
}
