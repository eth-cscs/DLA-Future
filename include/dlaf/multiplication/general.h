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

#include "dlaf/common/assert.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/multiplication/general/api.h"
#include "dlaf/util_matrix.h"

namespace dlaf::multiplication {

template <Device D, class T>
void generalSubMatrix(const SizeType a, const SizeType b, const blas::Op opA, const blas::Op opB,
                      const T alpha, Matrix<const T, D>& mat_a, Matrix<const T, D>& mat_b, const T beta,
                      Matrix<T, D>& mat_c) {
  DLAF_ASSERT(dlaf::matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(dlaf::matrix::square_blocksize(mat_b), mat_b);
  DLAF_ASSERT(dlaf::matrix::square_blocksize(mat_c), mat_c);

  // TODO check assertions. these are superflous
  DLAF_ASSERT(dlaf::matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(dlaf::matrix::square_size(mat_b), mat_b);
  DLAF_ASSERT(dlaf::matrix::square_size(mat_c), mat_c);

  const SizeType m = mat_a.nrTiles().rows();
  DLAF_ASSERT(a <= b, a, b);
  DLAF_ASSERT(a >= 0 and a < m, a, m);

  internal::GeneralSub<D, T>::call(a, b, opA, opB, alpha, mat_a, mat_b, beta, mat_c);
}

}
