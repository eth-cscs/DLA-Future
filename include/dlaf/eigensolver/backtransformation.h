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

#include "dlaf/communication/communicator_grid.h"
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
  DLAF_ASSERT(mat_c.size().rows() == mat_v.size().rows(), mat_c, mat_v);
  DLAF_ASSERT(mat_c.blockSize().rows() == mat_v.blockSize().rows(), mat_c, mat_v);

  const SizeType m = mat_v.size().rows();
  const SizeType mb = mat_v.blockSize().rows();
  SizeType nr_reflectors_blocks = std::max<SizeType>(0, util::ceilDiv(m - mb - 1, mb));
  DLAF_ASSERT(taus.size() == nr_reflectors_blocks, taus.size(), mat_v, nr_reflectors_blocks);

  internal::BackTransformation<backend, device, T>::call_FC(mat_c, mat_v, taus);
}

/// Eigenvalue back-transformation implementation on distributed memory.
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
/// @pre mat_c is distributed,
/// @pre mat_v is distributed according to grid.
template <Backend backend, Device device, class T>
void backTransformation(comm::CommunicatorGrid grid, Matrix<T, device>& mat_c,
                        Matrix<const T, device>& mat_v,
                        common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus) {
  DLAF_ASSERT(matrix::equal_process_grid(mat_c, grid), mat_c, grid);
  DLAF_ASSERT(matrix::equal_process_grid(mat_v, grid), mat_v, grid);
  DLAF_ASSERT(square_size(mat_v), mat_v);
  DLAF_ASSERT(square_blocksize(mat_v), mat_v);
  DLAF_ASSERT(mat_c.size().rows() == mat_v.size().rows(), mat_c, mat_v);
  DLAF_ASSERT(mat_c.blockSize().rows() == mat_v.blockSize().rows(), mat_c, mat_v);

  const SizeType m = mat_v.size().rows();
  const SizeType mb = mat_v.blockSize().rows();
  SizeType nr_reflectors_blocks = mat_v.distribution().template nextLocalTileFromGlobalTile<Coord::Col>(
      std::max<SizeType>(0, util::ceilDiv(m - mb - 1, mb)));
  DLAF_ASSERT(taus.size() == nr_reflectors_blocks, taus.size(), mat_v, nr_reflectors_blocks);

  internal::BackTransformation<backend, device, T>::call_FC(grid, mat_c, mat_v, taus);
}

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
void backTransformationT2B(matrix::Matrix<T, device>& mat_e, matrix::Matrix<const T, device>& mat_i) {
  DLAF_ASSERT(matrix::local_matrix(mat_e), mat_e);
  DLAF_ASSERT(matrix::local_matrix(mat_i), mat_i);

  DLAF_ASSERT(matrix::square_size(mat_i), mat_i);
  DLAF_ASSERT(matrix::square_blocksize(mat_i), mat_i);

  DLAF_ASSERT(mat_i.size().rows() == mat_e.size().rows(), mat_i, mat_e);
  DLAF_ASSERT(mat_i.blockSize().rows() == mat_e.blockSize().rows(), mat_i, mat_e);

  internal::BackTransformationT2B<backend, device, T>::call(mat_e, mat_i);
}
}
}
