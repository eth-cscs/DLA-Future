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
#include "dlaf/eigensolver/backtransformation/impl-t2b.h"
#include "dlaf/eigensolver/backtransformation/mc.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {

/// Eigenvalue back-transformation implementation on local memory.
///
/// It computes Q C, where Q = HH(1) HH(2) ... HH(m-mb)
/// (HH(j) is the House-Holder transformation (I - v tau vH)
/// defined by the j-th element of tau and the HH reflector stored in the j-th column of the matrix V.
///
/// @param mat_c contains the (m x n) matrix C (blocksize (mb x nb)), while on exit it contains Q C.
/// @param mat_v is (m x m) matrix with blocksize (mb x mb), which contains the Householder reflectors.
/// The j-th HH reflector is v_j = (1, V(mb + j : n, j)).
/// @param taus is a (blocked) vector of size m (blocksize mb). The j-th element is the scaling factor
/// for the j-th HH tranformation.
/// @pre mat_c is not distributed,
/// @pre mat_v is not distributed.
template <Backend backend, Device device, class T>
void backTransformation(Matrix<T, device>& mat_c, Matrix<const T, device>& mat_v,
                        common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus) {
  DLAF_ASSERT(matrix::local_matrix(mat_c), mat_c);
  DLAF_ASSERT(matrix::local_matrix(mat_v), mat_v);
  DLAF_ASSERT(square_size(mat_v), mat_v);
  DLAF_ASSERT(square_blocksize(mat_v), mat_v);
  DLAF_ASSERT(mat_c.blockSize().rows() == mat_v.blockSize().rows(), mat_c, mat_v);

  const SizeType m = mat_v.size().rows();
  const SizeType mb = mat_v.blockSize().rows();
  SizeType nr_reflectors_blocks = std::max<SizeType>(0, util::ceilDiv(m - mb - 1, mb));
  DLAF_ASSERT(taus.size() == nr_reflectors_blocks, taus.size(), mat_v, nr_reflectors_blocks);

  internal::BackTransformation<backend, device, T>::call_FC(mat_c, mat_v, taus);
}

// TODO DOC
template <Backend backend, Device device, class T>
void backTransformationT2B(matrix::Matrix<T, device>& mat_e, matrix::Matrix<const T, device>& mat_v,
                           matrix::Matrix<const T, device>& mat_taus) {
  // TODO check conditions
  DLAF_ASSERT(matrix::local_matrix(mat_e), mat_e);
  DLAF_ASSERT(matrix::local_matrix(mat_v), mat_v);

  DLAF_ASSERT(matrix::square_size(mat_v), mat_v);
  DLAF_ASSERT(matrix::square_blocksize(mat_v), mat_v);

  DLAF_ASSERT(matrix::equal_size(mat_v, mat_taus), mat_v, mat_taus);
  DLAF_ASSERT(matrix::equal_blocksize(mat_v, mat_taus), mat_v, mat_taus);

  // TODO check taus with respect to v

  internal::BackTransformationT2B<backend, device, T>::call(mat_e, mat_v, mat_taus);
}
}
}
