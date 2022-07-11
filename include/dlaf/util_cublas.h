//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#ifdef DLAF_WITH_GPU

#include <blas.hh>
#include "dlaf/gpu/blas/api.h"
#include "dlaf/util_cuda.h"

namespace dlaf {
namespace util {

template <typename T>
constexpr auto blasToCublasCast(T x) {
  return cppToCudaCast(x);
}

inline constexpr cublasSideMode_t blasToCublas(const blas::Side side) {
  switch (side) {
    case blas::Side::Left:
      return CUBLAS_SIDE_LEFT;
    case blas::Side::Right:
      return CUBLAS_SIDE_RIGHT;
    default:
      return {};
  }
}

inline constexpr cublasFillMode_t blasToCublas(const blas::Uplo uplo) {
  switch (uplo) {
    case blas::Uplo::Lower:
      return CUBLAS_FILL_MODE_LOWER;
    case blas::Uplo::Upper:
      return CUBLAS_FILL_MODE_UPPER;
    case blas::Uplo::General:
      return CUBLAS_FILL_MODE_FULL;
    default:
      return {};
  }
}

/// Predicate returning true for all coordinates lying in the lower part or on the diagonal of a matrix
__device__ inline constexpr bool isLower(unsigned i, unsigned j) noexcept {
  return i >= j;
}

/// Predicate returning true for all coordinates lying in the upper part or on the diagonal of a matrix
__device__ inline constexpr bool isUpper(unsigned i, unsigned j) noexcept {
  return i <= j;
}

/// Predicate returning true for all coordinates valid for a "general" matrix (i.e. always true)
__device__ inline constexpr bool isGeneral(unsigned, unsigned) noexcept {
  return true;
}

inline constexpr cublasOperation_t blasToCublas(const blas::Op op) {
  switch (op) {
    case blas::Op::NoTrans:
      return CUBLAS_OP_N;
    case blas::Op::Trans:
      return CUBLAS_OP_T;
    case blas::Op::ConjTrans:
      return CUBLAS_OP_C;
    default:
      return {};
  }
}

inline constexpr cublasDiagType_t blasToCublas(const blas::Diag diag) {
  switch (diag) {
    case blas::Diag::Unit:
      return CUBLAS_DIAG_UNIT;
    case blas::Diag::NonUnit:
      return CUBLAS_DIAG_NON_UNIT;
    default:
      return {};
  }
}

}
}

#endif
