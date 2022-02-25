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

#include <blas.hh>
#include <lapack/util.hh>
#include "dlaf/common/assert.h"

namespace lapack {

inline MatrixType uploToMatrixType(blas::Uplo uplo) {
  switch (uplo) {
    case blas::Uplo::Lower:
      return MatrixType::Lower;
    case blas::Uplo::Upper:
      return MatrixType::Upper;
    case blas::Uplo::General:
      return MatrixType::General;
    default:
      return DLAF_UNREACHABLE(MatrixType);
  }
}

/// Overload of lacpy with blas::Uplo instead of lapack::MatrixType
template <class T>
void lacpy(blas::Uplo uplo, int64_t m, int64_t n, const T* a, int64_t lda, T* b, int64_t ldb) {
  lacpy(uploToMatrixType(uplo), m, n, a, lda, b, ldb);
}

/// Overload of laset with blas::Uplo instead of lapack::MatrixType
template <class T>
void laset(blas::Uplo uplo, int64_t m, int64_t n, T alpha, T beta, T* a, int64_t lda) {
  laset(uploToMatrixType(uplo), m, n, alpha, beta, a, lda);
}

}
