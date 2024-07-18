//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <dlaf/eigensolver/bt_band_to_tridiag/api.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_ref.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace dlaf::eigensolver::internal {

/// Eigenvalue back-transformation implementation on local memory, which applies the inverse of the
/// transformation used to get a tridiagonal matrix from a band one.
///
/// It computes E -= V T V* E, applying to a general matrix E the inverse of the transformation described
/// by the reflectors in V (block-wise, so T represents the T factor which embeds the information about
/// taus), which are the ones used to transform a band matrix to a tridiagonal matrix.
///
/// In particular, V and T are obtained using data about reflectors and taus passed via @p mat_hh
/// where they are stored using following compact representation
///
/// compact           extended
/// AT BT CT DT       1  0  0  0
/// A1 B1 C1 D1       A1 1  0  0
/// A2 B2 C2 D2       A2 B1 1  0
/// A3 B3 C3 D3       A3 B2 C1 1
///                   0  B3 C2 D1
///                   0  0  C3 D2
///                   0  0  0  D3
///
/// where A, B, C and D refers to distinct reflectors, with their components numbered and their taus
/// identified by the letter T.
///
/// @param mat_hh matrix containing reflectors together with taus (compact form see representation above)
/// @pre @p mat_hh is not distributed
/// @pre @p mat_hh has size (N x N)
/// @pre @p mat_hh has blocksize (NB x NB)
/// @pre @p mat_hh has tilesize (NB x NB)
///
/// @param mat_e matrix to which the inverse transformation is applied to
/// @pre @p mat_e is not distributed
/// @pre @p mat_e has size (N x M)
/// @pre @p mat_e has blocksize (NB x MB)
/// @pre @p mat_e has tilesize (NB x MB)
///
/// @param band_size size of the reflectors (normal one, not constrained by any matrix size limit)
/// @pre @p band_size is a divisor of `mat_hh.blockSize().cols()`
template <Backend B, Device D, class T>
void bt_band_to_tridiagonal(const SizeType band_size, matrix::internal::MatrixRef<T, D>& mat_e,
                            matrix::Matrix<const T, Device::CPU>& mat_hh) {
  DLAF_ASSERT(matrix::local_matrix(mat_e), mat_e);
  DLAF_ASSERT(matrix::local_matrix(mat_hh), mat_hh);

  DLAF_ASSERT(matrix::square_size(mat_hh), mat_hh);
  DLAF_ASSERT(matrix::square_blocksize(mat_hh), mat_hh);

  DLAF_ASSERT(mat_hh.size().rows() == mat_e.size().rows(), mat_hh, mat_e);
  DLAF_ASSERT(mat_hh.blockSize().rows() == mat_e.blockSize().rows(), mat_hh, mat_e);

  DLAF_ASSERT(matrix::single_tile_per_block(mat_e), mat_e);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_hh), mat_hh);

  DLAF_ASSERT(band_size >= 2, band_size);
  DLAF_ASSERT(mat_hh.blockSize().rows() % band_size == 0, mat_hh.blockSize(), band_size);

  BackTransformationT2B<B, D, T>::call(band_size, mat_e, mat_hh);
}

/// Eigenvalue back-transformation implementation, which applies the inverse of the transformation used
/// to get a tridiagonal matrix from a band one.
///
/// It computes E -= V T V* E, applying to a general matrix E the inverse of the transformation described
/// by the reflectors in V (block-wise, so T represents the T factor which embeds the information about
/// taus), which are the ones used to transform a band matrix to a tridiagonal matrix.
///
/// In particular, V and T are obtained using data about reflectors and taus passed via @p mat_hh
/// where they are stored using following compact representation
///
/// compact           extended
/// AT BT CT DT       1  0  0  0
/// A1 B1 C1 D1       A1 1  0  0
/// A2 B2 C2 D2       A2 B1 1  0
/// A3 B3 C3 D3       A3 B2 C1 1
///                   0  B3 C2 D1
///                   0  0  C3 D2
///                   0  0  0  D3
///
/// where A, B, C and D refers to distinct reflectors, with their components numbered and their taus
/// identified by the letter T.
///
/// @param mat_hh matrix containing reflectors together with taus (compact form see representation above)
/// @pre @p mat_hh is distributed according to @p grid
/// @pre @p mat_hh has size (N x N)
/// @pre @p mat_hh has blocksize (NB x NB)
/// @pre @p mat_hh has tilesize (NB x NB)
///
/// @param mat_e matrix to which the inverse transformation is applied to
/// @pre @p mat_e is distributed according to @p grid
/// @pre @p mat_e has size (N x M)
/// @pre @p mat_e has blocksize (NB x MB)
/// @pre @p mat_e has tilesize (NB x MB)
///
/// @param band_size size of the reflectors (normal one, not constrained by any matrix size limit)
/// @pre @p band_size is a divisor of `mat_hh.blockSize().cols()`
template <Backend B, Device D, class T>
void bt_band_to_tridiagonal(comm::CommunicatorGrid& grid, const SizeType band_size,
                            matrix::internal::MatrixRef<T, D>& mat_e,
                            Matrix<const T, Device::CPU>& mat_hh) {
  DLAF_ASSERT(matrix::equal_process_grid(mat_e, grid), mat_e, grid);
  DLAF_ASSERT(matrix::equal_process_grid(mat_hh, grid), mat_hh, grid);

  DLAF_ASSERT(matrix::square_size(mat_hh), mat_hh);
  DLAF_ASSERT(matrix::square_blocksize(mat_hh), mat_hh);

  DLAF_ASSERT(mat_hh.size().rows() == mat_e.size().rows(), mat_hh, mat_e);
  DLAF_ASSERT(mat_hh.blockSize().rows() == mat_e.blockSize().rows(), mat_hh, mat_e);

  DLAF_ASSERT(band_size >= 2, band_size);
  DLAF_ASSERT(mat_hh.blockSize().rows() % band_size == 0, mat_hh.blockSize(), band_size);

  BackTransformationT2B<B, D, T>::call(grid, band_size, mat_e, mat_hh);
}
}
