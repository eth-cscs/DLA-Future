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
/// @brief Triangular Solve implementation on local memory, solving the linear system X = alpha A^(-1) B.
/// In the specific case of a Hermitian positive definite generalized eigenvalue problem,
/// B is a matrix of the eigenvectors of the reduced phase,
/// and A is a matrix obtained from a Cholesky decomposition.
///
/// @param side specifies whether op(A) appears on the \a Left or on the \a Right of dlaf::Matrix A
/// @param uplo specifies whether the dlaf::Matrix A is a \a Lower or \a Upper triangular matrix
/// @param op specifies the form of op(A) to be used in the matrix multiplication: \a NoTrans, \a Trans,
/// \a ConjTrans
/// @param diag specifies whether dlaf::Matrix is \a Unit or not (\a NonUnit) triangular
/// @param alpha specifies the scalar alpha
/// @tparam A refers to a triangular dlaf::Matrix object
/// @tparam B refers to a dlaf::Matrix object
template <class T>
void triangular_solve(blas::Side side, blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha,
                      Matrix<T, Device::CPU>& A, Matrix<T, Device::CPU>& B) {
  // Set up executor on the default queue with high priority.
  hpx::threads::scheduled_executor executor_hp =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_high);

  // Set up executor on the default queue with default priority.
  hpx::threads::scheduled_executor executor_normal =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_default);

  // Check if matrix A is square
  util_matrix::assertSizeSquare(A, "TriangularSolve", "A");
  // Check if block matrix A is square
  util_matrix::assertBlocksizeSquare(A, "TriangularSolve", "A");
  // Check if A and B dimensions are compatible
  util_matrix::assertMultipliableMatrices(A, B, side, op, "TriangularSolve", "A", "B");
  // Check if matrix A is stored on local memory
  util_matrix::assertLocalMatrix(A, "TriangularSolve", "A");
  // Check if matrix B is stored on local memory
  util_matrix::assertLocalMatrix(B, "TriangularSolve", "B");

  SizeType m = B.nrTiles().rows();
  SizeType n = B.nrTiles().cols();

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
                          diag, alpha, A.read(LocalTileIndex{k, k}), std::move(B(kj)));

            for (SizeType i = k - 1; i > -1; --i) {
              // Choose queue priority
              auto trailing_executor = (i == k - 1) ? executor_hp : executor_normal;

              auto beta = static_cast<T>(-1.0) / alpha;
              // Matrix multiplication to update other eigenvectors
              hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), op,
                            blas::Op::NoTrans, beta, A.read(LocalTileIndex{i, k}), B.read(kj), 1.0,
                            std::move(B(LocalTileIndex{i, j})));
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
                          diag, alpha, A.read(LocalTileIndex{k, k}), std::move(B(kj)));

            for (SizeType i = k + 1; i < m; ++i) {
              // Choose queue priority
              auto trailing_executor = (i == k + 1) ? executor_hp : executor_normal;

              auto beta = static_cast<T>(-1.0) / alpha;
              // Matrix multiplication to update other eigenvectors
              hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), op,
                            blas::Op::NoTrans, beta, A.read(LocalTileIndex{k, i}), B.read(kj), 1.0,
                            std::move(B(LocalTileIndex{i, j})));
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
                          diag, alpha, A.read(LocalTileIndex{k, k}), std::move(B(ik)));

            for (SizeType j = k + 1; j < n; ++j) {
              // Choose queue priority
              auto trailing_executor = (j == k - 1) ? executor_hp : executor_normal;

              auto beta = static_cast<T>(-1.0) / alpha;
              // Matrix multiplication to update other eigenvectors
              hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>),
                            blas::Op::NoTrans, op, beta, B.read(ik), A.read(LocalTileIndex{k, j}), 1.0,
                            std::move(B(LocalTileIndex{i, j})));
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
                          diag, alpha, A.read(LocalTileIndex{k, k}), std::move(B(ik)));

            for (SizeType j = k - 1; j > -1; --j) {
              // Choose queue priority
              auto trailing_executor = (j == k - 1) ? executor_hp : executor_normal;

              auto beta = static_cast<T>(-1.0) / alpha;
              // Matrix multiplication to update other eigenvectors
              hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>),
                            blas::Op::NoTrans, op, beta, B.read(ik), A.read(LocalTileIndex{j, k}), 1.0,
                            std::move(B(LocalTileIndex{i, j})));
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
                          diag, alpha, A.read(LocalTileIndex{k, k}), std::move(B(kj)));

            for (SizeType i = k + 1; i < m; ++i) {
              // Choose queue priority
              auto trailing_executor = (i == k + 1) ? executor_hp : executor_normal;

              auto beta = static_cast<T>(-1.0) / alpha;
              // Matrix multiplication to update other eigenvectors
              hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), op,
                            blas::Op::NoTrans, beta, A.read(LocalTileIndex{i, k}), B.read(kj), 1.0,
                            std::move(B(LocalTileIndex{i, j})));
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
                          diag, alpha, A.read(LocalTileIndex{k, k}), std::move(B(kj)));

            for (SizeType i = k - 1; i > -1; --i) {
              // Choose queue priority
              auto trailing_executor = (i == k - 1) ? executor_hp : executor_normal;

              auto beta = static_cast<T>(-1.0) / alpha;
              // Matrix multiplication to update other eigenvectors
              hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), op,
                            blas::Op::NoTrans, beta, A.read(LocalTileIndex{k, i}), B.read(kj), 1.0,
                            std::move(B(LocalTileIndex{i, j})));
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
                          diag, alpha, A.read(LocalTileIndex{k, k}), std::move(B(ik)));

            for (SizeType j = k - 1; j > -1; --j) {
              // Choose queue priority
              auto trailing_executor = (j == k - 1) ? executor_hp : executor_normal;

              auto beta = static_cast<T>(-1.0) / alpha;
              // Matrix multiplication to update other eigenvectors
              hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>),
                            blas::Op::NoTrans, op, beta, B.read(ik), A.read(LocalTileIndex{k, j}), 1.0,
                            std::move(B(LocalTileIndex{i, j})));
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
                          diag, alpha, A.read(LocalTileIndex{k, k}), std::move(B(ik)));

            for (SizeType j = k + 1; j < n; ++j) {
              // Choose queue priority
              auto trailing_executor = (j == k - 1) ? executor_hp : executor_normal;

              auto beta = static_cast<T>(-1.0) / alpha;
              // Matrix multiplication to update other eigenvectors
              hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>),
                            blas::Op::NoTrans, op, beta, B.read(ik), A.read(LocalTileIndex{j, k}), 1.0,
                            std::move(B(LocalTileIndex{i, j})));
            }
          }
        }
      }
    }
  }
}
}
