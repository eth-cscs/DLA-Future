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

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix.h"
#include "dlaf/solver/triangular/mc/triangular_LLN.h"
#include "dlaf/solver/triangular/mc/triangular_LLT.h"
#include "dlaf/solver/triangular/mc/triangular_LUN.h"
#include "dlaf/solver/triangular/mc/triangular_LUT.h"
#include "dlaf/solver/triangular/mc/triangular_RLN.h"
#include "dlaf/solver/triangular/mc/triangular_RLT.h"
#include "dlaf/solver/triangular/mc/triangular_RUN.h"
#include "dlaf/solver/triangular/mc/triangular_RUT.h"
#include "dlaf/util_matrix.h"

namespace dlaf {

template <class T>
void Solver<Backend::MC>::triangular(blas::Side side, blas::Uplo uplo, blas::Op op, blas::Diag diag,
                                     T alpha, Matrix<const T, Device::CPU>& mat_a,
                                     Matrix<T, Device::CPU>& mat_b) {
  // Check if matrix A is square
  util_matrix::assertSizeSquare(mat_a, "TriangularSolver", "mat_a");
  // Check if block matrix A is square
  util_matrix::assertBlocksizeSquare(mat_a, "TriangularSolver", "mat_a");
  // Check if A and B dimensions are compatible
  util_matrix::assertMultipliableMatrices(mat_a, mat_b, side, op, "TriangularSolver", "mat_a", "mat_b");
  // Check if matrix A is stored on local memory
  util_matrix::assertLocalMatrix(mat_a, "TriangularSolver", "mat_a");
  // Check if matrix B is stored on local memory
  util_matrix::assertLocalMatrix(mat_b, "TriangularSolver", "mat_b");

  if (side == blas::Side::Left) {
    if (uplo == blas::Uplo::Lower) {
      if (op == blas::Op::NoTrans) {
        // Left Lower NoTrans
        internal::mc::triangular_LLN(diag, alpha, mat_a, mat_b);
      }
      else {
        // Left Lower Trans/ConjTrans
        internal::mc::triangular_LLT(op, diag, alpha, mat_a, mat_b);
      }
    }
    else {
      if (op == blas::Op::NoTrans) {
        // Left Upper NoTrans
        internal::mc::triangular_LUN(diag, alpha, mat_a, mat_b);
      }
      else {
        // Left Upper Trans/ConjTrans
        internal::mc::triangular_LUT(op, diag, alpha, mat_a, mat_b);
      }
    }
  }
  else {
    if (uplo == blas::Uplo::Lower) {
      if (op == blas::Op::NoTrans) {
        // Right Lower NoTrans
        internal::mc::triangular_RLN(diag, alpha, mat_a, mat_b);
      }
      else {
        // Right Lower Trans/ConjTrans
        internal::mc::triangular_RLT(op, diag, alpha, mat_a, mat_b);
      }
    }
    else {
      if (op == blas::Op::NoTrans) {
        // Right Upper NoTrans
        internal::mc::triangular_RUN(diag, alpha, mat_a, mat_b);
      }
      else {
        // Right Upper Trans/ConjTrans
        internal::mc::triangular_RUT(op, diag, alpha, mat_a, mat_b);
      }
    }
  }
}

template <class T>
void Solver<Backend::MC>::triangular(comm::CommunicatorGrid grid, blas::Side side, blas::Uplo uplo,
                                     blas::Op op, blas::Diag diag, T alpha,
                                     Matrix<const T, Device::CPU>& mat_a,
                                     Matrix<T, Device::CPU>& mat_b) {
  // Check if matrix A is square
  util_matrix::assertSizeSquare(mat_a, "TriangularSolver", "mat_a");
  // Check if block matrix A is square
  util_matrix::assertBlocksizeSquare(mat_a, "TriangularSolver", "mat_a");
  // Check if A and B dimensions are compatible
  util_matrix::assertMultipliableMatrices(mat_a, mat_b, side, op, "TriangularSolver", "mat_a", "mat_b");
  // Check compatibility of the communicator grid and the distribution of matrix A
  util_matrix::assertMatrixDistributedOnGrid(grid, mat_a, "TriangularSolver", "mat_a", "grid");
  // Check compatibility of the communicator grid and the distribution of matrix B
  util_matrix::assertMatrixDistributedOnGrid(grid, mat_b, "TriangularSolver", "mat_b", "grid");

  if (side == blas::Side::Left) {
    if (uplo == blas::Uplo::Lower) {
      if (op == blas::Op::NoTrans) {
        // Left Lower NoTrans
        internal::mc::triangular_LLN(grid, diag, alpha, mat_a, mat_b);
      }
      else {
        // Left Lower Trans/ConjTrans
        throw std::runtime_error("Distributed Left Lower Trans/ConjTrans case not yet implemented");
      }
    }
    else {
      if (op == blas::Op::NoTrans) {
        // Left Upper NoTrans
        throw std::runtime_error("Distributed Left Upper NoTrans case not yet implemented");
      }
      else {
        // Left Upper Trans/ConjTrans
        throw std::runtime_error("Distributed Left Upper Trans/ConjTrans case not yet implemented");
      }
    }
  }
  else {
    if (uplo == blas::Uplo::Lower) {
      if (op == blas::Op::NoTrans) {
        // Right Lower NoTrans
        throw std::runtime_error("Distributed Right Lower NoTrans case not yet implemented");
      }
      else {
        // Right Lower Trans/ConjTrans
        throw std::runtime_error("Distributed Right Lower Trans/ConjTrans case not yet implemented");
      }
    }
    else {
      if (op == blas::Op::NoTrans) {
        // Right Upper NoTrans
        throw std::runtime_error("Distributed Right Upper NoTrans case not yet implemented");
      }
      else {
        // Right Upper Trans/ConjTrans
        throw std::runtime_error("Distributed Right Upper Trans/ConjTrans case not yet implemented");
      }
    }
  }
}

}
