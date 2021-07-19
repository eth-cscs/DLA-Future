//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <blas.hh>
#include "dlaf/eigensolver/backtransformation/mc.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {

/// Eigenvalue back-transformation implementation on local memory.
///
/// It computes Q C, where Q = HH(1) HH(2) ... HH(m-b) (HH(j) is the House-Holder transformation defined
/// by the j-th element of tau and the HH reflector stored in the j-th column.
///
/// The HH reflector are computed using the @see computeTFactor() method, starting from a matrix
/// containing the elementary reflectors (@param mat_v) and an array of taus, associated with each
/// reflector (@param taus). For a generic matrix @param mat_c (mxn), @param mat_v has size (mxm) and
/// @param taus corresponds to the total number of reflectors (i.e. the number of non zero columns of
/// @param mat_v).
///
/// Each tau is computed selecting a column of @param mat_v (called @p v). In case of real number, tau =
/// 2 / (vH v), while in the complex case the real part of tau corresponds to  [1 + sqrt(1 - vH v
/// taui^2)]/(vH v), where @p taui is complex part (random value).
///
/// @param mat_c contains the matrix C, while on exit it contains Q C.
/// @param mat_v is a lower triangular matrix, containing Householder vectors (reflectors).
/// @param taus is a vectors of scalar, associated with the related elementary reflector.
/// @pre mat_c is not distributed,
/// @pre mat_v is not distributed.
template <Backend backend, Device device, class T>
void backTransformation(Matrix<T, device>& mat_c, Matrix<const T, device>& mat_v,
                        common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus) {
  DLAF_ASSERT(matrix::local_matrix(mat_c), mat_c);
  DLAF_ASSERT(matrix::local_matrix(mat_v), mat_v);
  DLAF_ASSERT(mat_c.blockSize().rows() == mat_v.blockSize().rows(), mat_c, mat_v);

  internal::BackTransformation<backend, device, T>::call_FC(mat_c, mat_v, taus);
}

}
}
