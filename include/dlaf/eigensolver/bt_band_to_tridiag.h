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

#include "dlaf/eigensolver/bt_band_to_tridiag/mc.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf::eigensolver {

// Eigenvalue back-transformation implementation on local memory, which applies the inverse of the
// transformation used to get a tridiagonal matrix from band one.
//
// It computes E -= V T V* E, applying to a general matrix E the inverse of the transformation described
// by the reflectors in V (block-wise, so T represents the T factor which embeds the information about
// taus), which are the ones used to transform a band matrix to a tridiagonal matrix.
//
// In particular, V and T are obatined using data about reflectors and taus passed via @p mat_i
// where they are stored using following compact representation
//
// compact           extended
// AT BT CT DT       1  0  0  0
// A1 B1 C1 D1       A1 1  0  0
// A2 B2 C2 D2       A2 B1 1  0
// A3 B3 C3 D3       A3 B2 C1 1
//                   0  B3 C2 D1
//                   0  0  C3 D2
//                   0  0  0  D3
//
// where A, B, C and D refers to distinct reflectors, with their components numbered and their taus
// identified by the letter T.
//
// @param mat_i matrix containing reflectors together with taus (compact form see representation above)
// @param mat_e matrix to which the inverse transformation is applied to
template <Backend backend, Device device, class T>
void backTransformationBandToTridiag(matrix::Matrix<T, device>& mat_e,
                                     matrix::Matrix<const T, device>& mat_i) {
  DLAF_ASSERT(matrix::local_matrix(mat_e), mat_e);
  DLAF_ASSERT(matrix::local_matrix(mat_i), mat_i);

  DLAF_ASSERT(matrix::square_size(mat_i), mat_i);
  DLAF_ASSERT(matrix::square_blocksize(mat_i), mat_i);

  DLAF_ASSERT(mat_i.size().rows() == mat_e.size().rows(), mat_i, mat_e);
  DLAF_ASSERT(mat_i.blockSize().rows() == mat_e.blockSize().rows(), mat_i, mat_e);

  internal::BackTransformationT2B<backend, device, T>::call(mat_e, mat_i);
}

}
