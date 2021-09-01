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
#include "dlaf/eigensolver/gen_eigensolver/impl.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {

/// TODO
template <Backend backend, Device device, class T>
ReturnEigensolverType<T, device> genEigensolver(blas::Uplo uplo, Matrix<T, device>& mat_a,
                                                Matrix<T, device>& mat_b) {
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);
  DLAF_ASSERT(matrix::local_matrix(mat_b), mat_b);
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_size(mat_b), mat_b);
  DLAF_ASSERT(matrix::square_blocksize(mat_b), mat_b);
  DLAF_ASSERT(mat_a.size() == mat_b.size(), mat_a, mat_b);
  DLAF_ASSERT(mat_a.blockSize() == mat_b.blockSize(), mat_a, mat_b);

  return internal::GenEigensolver<backend, device, T>::call(uplo, mat_a, mat_b);
}
}
}
