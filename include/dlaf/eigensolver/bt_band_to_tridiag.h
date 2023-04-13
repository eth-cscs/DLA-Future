//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "dlaf/eigensolver/bt_band_to_tridiag/api.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf::eigensolver {

// Eigenvalue back-transformation implementation on local memory, which applies the inverse of the
// transformation used to get a tridiagonal matrix from a band one.
//
// It computes E -= V T V* E, applying to a general matrix E the inverse of the transformation described
// by the reflectors in V (block-wise, so T represents the T factor which embeds the information about
// taus), which are the ones used to transform a band matrix to a tridiagonal matrix.
//
// In particular, V and T are obtained using data about reflectors and taus passed via @p mat_hh
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
// @param mat_hh matrix containing reflectors together with taus (compact form see representation above)
// @param mat_e matrix to which the inverse transformation is applied to
// @param band_size size of the reflectors (normal one, not constrained by any matrix size limit)
// @pre mat_hh has a square size
// @pre mat_hh has a square block size
// @pre mat_e and mat_hh share the same number of rows
// @pre mat_e block size and mat_hh block size share the same number of rows
// @pre band_size is a divisor of mat_hh.blockSize().cols()
// @pre mat_e is not distributed
// @pre mat_hh is not distributed
template <Backend B, Device D, class T>
void backTransformationBandToTridiag(const SizeType band_size, matrix::Matrix<T, D>& mat_e,
                                     matrix::Matrix<const T, Device::CPU>& mat_hh) {
  DLAF_ASSERT(matrix::local_matrix(mat_e), mat_e);
  DLAF_ASSERT(matrix::local_matrix(mat_hh), mat_hh);

  DLAF_ASSERT(matrix::square_size(mat_hh), mat_hh);
  DLAF_ASSERT(matrix::square_blocksize(mat_hh), mat_hh);

  DLAF_ASSERT(mat_hh.size().rows() == mat_e.size().rows(), mat_hh, mat_e);
  DLAF_ASSERT(mat_hh.blockSize().rows() == mat_e.blockSize().rows(), mat_hh, mat_e);

  DLAF_ASSERT(band_size >= 2, band_size);
  DLAF_ASSERT(mat_hh.blockSize().rows() % band_size == 0, mat_hh.blockSize(), band_size);

  internal::BackTransformationT2B<B, D, T>::call(band_size, mat_e, mat_hh);
}

template <Backend B, Device D, class T>
void backTransformationBandToTridiag(comm::CommunicatorGrid grid, const SizeType band_size,
                                     matrix::Matrix<T, D>& mat_e,
                                     matrix::Matrix<const T, Device::CPU>& mat_hh) {
  DLAF_ASSERT(matrix::equal_process_grid(mat_e, grid), mat_e, grid);
  DLAF_ASSERT(matrix::equal_process_grid(mat_hh, grid), mat_hh, grid);

  DLAF_ASSERT(matrix::square_size(mat_hh), mat_hh);
  DLAF_ASSERT(matrix::square_blocksize(mat_hh), mat_hh);

  DLAF_ASSERT(mat_hh.size().rows() == mat_e.size().rows(), mat_hh, mat_e);
  DLAF_ASSERT(mat_hh.blockSize().rows() == mat_e.blockSize().rows(), mat_hh, mat_e);

  DLAF_ASSERT(band_size >= 2, band_size);
  DLAF_ASSERT(mat_hh.blockSize().rows() % band_size == 0, mat_hh.blockSize(), band_size);

  internal::BackTransformationT2B<B, D, T>::call(grid, band_size, mat_e, mat_hh);
}
}
