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
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);
  DLAF_ASSERT(matrix::local_matrix(mat_b), mat_b);

  if (side == blas::Side::Left) {
    // Check if A and B dimensions are compatible
    DLAF_ASSERT(matrix::multipliable(mat_a, mat_b, mat_b, op, blas::Op::NoTrans), mat_a, mat_b, op);

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
    // Check if A and B dimensions are compatible
    DLAF_ASSERT(matrix::multipliable(mat_b, mat_a, mat_b, blas::Op::NoTrans, op), mat_a, mat_b, op);

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
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);
  DLAF_ASSERT(matrix::equal_process_grid(mat_b, grid), mat_b, grid);

  if (side == blas::Side::Left) {
    // Check if A and B dimensions are compatible
    DLAF_ASSERT(matrix::multipliable(mat_a, mat_b, mat_b, op, blas::Op::NoTrans), mat_a, mat_b, op);

    if (uplo == blas::Uplo::Lower) {
      if (op == blas::Op::NoTrans) {
        // Left Lower NoTrans
        internal::mc::triangular_LLN(grid, diag, alpha, mat_a, mat_b);
      }
      else {
        // Left Lower Trans/ConjTrans
        std::cout << "Distributed Left Lower Trans/ConjTrans case not yet implemented" << std::endl;
        std::abort();
      }
    }
    else {
      if (op == blas::Op::NoTrans) {
        // Left Upper NoTrans
        std::cout << "Distributed Left Upper NoTrans case not yet implemented" << std::endl;
        std::abort();
      }
      else {
        // Left Upper Trans/ConjTrans
        std::cout << "Distributed Left Upper Trans/ConjTrans case not yet implemented" << std::endl;
        std::abort();
      }
    }
  }
  else {
    // Check if A and B dimensions are compatible
    DLAF_ASSERT(matrix::multipliable(mat_a, mat_b, mat_b, blas::Op::NoTrans, op), mat_a, mat_b, op);

    if (uplo == blas::Uplo::Lower) {
      if (op == blas::Op::NoTrans) {
        // Right Lower NoTrans
        std::cout << "Distributed Right Lower NoTrans case not yet implemented" << std::endl;
        std::abort();
      }
      else {
        // Right Lower Trans/ConjTrans
        std::cout << "Distributed Right Lower Trans/ConjTrans case not yet implemented" << std::endl;
        std::abort();
      }
    }
    else {
      if (op == blas::Op::NoTrans) {
        // Right Upper NoTrans
        std::cout << "Distributed Right Upper NoTrans case not yet implemented" << std::endl;
        std::abort();
      }
      else {
        // Right Upper Trans/ConjTrans
        std::cout << "Distributed Right Upper Trans/ConjTrans case not yet implemented" << std::endl;
        std::abort();
      }
    }
  }
}
}
