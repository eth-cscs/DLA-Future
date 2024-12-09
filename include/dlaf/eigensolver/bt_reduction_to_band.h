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

#include <blas.hh>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/eigensolver/bt_reduction_to_band/api.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_ref.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace dlaf::eigensolver::internal {

/// Eigenvalue back-transformation implementation on local memory.
///
/// It computes Q C, where Q = HH(1) HH(2) ... HH(m-b)
/// (HH(j) is the House-Holder transformation (I - v tau vH)
/// defined by the j-th element of tau and the HH reflector stored in the j-th column of the matrix V.
///
/// @param mat_c contains the (m x n) matrix C (blocksize (mb x nb)), while on exit it contains Q C.
/// @pre @p mat_c is not distributed
/// @pre @p mat_c has blocksize (NB x NB)
/// @pre @p mat_c has tilesize (NB x NB)
///
/// @param mat_v is (m x m) matrix with blocksize (mb x mb), which contains the Householder reflectors.
/// The j-th HH reflector is v_j = (1, V(mb + j : n, j)).
/// @pre @p mat_v is not distributed
/// @pre @p mat_v has blocksize (NB x NB)
/// @pre @p mat_v has tilesize (NB x NB)
///
/// @param mat_taus is the tau vector as returned by reductionToBand. The j-th element is the scaling
/// factor for the j-th HH tranformation.
template <Backend backend, Device device, class T>
void bt_reduction_to_band(const SizeType b, MatrixRef<T, device>& mat_c, Matrix<const T, device>& mat_v,
                          Matrix<const T, Device::CPU>& mat_taus) {
  DLAF_ASSERT(matrix::local_matrix(mat_c), mat_c);
  DLAF_ASSERT(matrix::local_matrix(mat_v), mat_v);
  DLAF_ASSERT(square_size(mat_v), mat_v);
  DLAF_ASSERT(square_blocksize(mat_v), mat_v);
  DLAF_ASSERT(mat_c.size().rows() == mat_v.size().rows(), mat_c, mat_v);
  DLAF_ASSERT(mat_c.blockSize().rows() == mat_v.blockSize().rows(), mat_c, mat_v);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_c), mat_c);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_v), mat_v);

  [[maybe_unused]] auto nr_reflectors_blocks = [&b, &mat_v]() {
    const SizeType m = mat_v.size().rows();
    const SizeType mb = mat_v.blockSize().rows();
    return std::max<SizeType>(0, util::ceilDiv(m - b - 1, mb));
  };
  DLAF_ASSERT(mat_taus.nrTiles().rows() == nr_reflectors_blocks(), mat_taus.size().rows(), mat_v, b);

  BackTransformationReductionToBand<backend, device, T>::call(b, mat_c, mat_v, mat_taus);
}

/// Eigenvalue back-transformation implementation on distributed memory.
///
/// It computes Q C, where Q = HH(1) HH(2) ... HH(m-mb)
/// (HH(j) is the House-Holder transformation (I - v tau vH)
/// defined by the j-th element of tau and the HH reflector stored in the j-th column of the matrix V.
///
/// @param mat_c contains the (m x n) matrix C (blocksize (mb x nb)), while on exit it contains Q C
/// @pre @p mat_c is distributed according to @p grid
/// @pre @p mat_c has blocksize (NB x NB)
/// @pre @p mat_c has tilesize (NB x NB)
///
/// @param mat_v is (m x m) matrix with blocksize (mb x mb), which contains the Householder reflectors.
/// The j-th HH reflector is v_j = (1, V(mb + j : n, j)).
/// @pre @p mat_v is distributed according to @p grid
/// @pre @p mat_v has blocksize (NB x NB)
/// @pre @p mat_v has tilesize (NB x NB)
///
/// @param mat_taus is the tau vector as returned by reductionToBand. The j-th element is the scaling
/// factor for the j-th HH tranformation.
template <Backend backend, Device device, class T>
void bt_reduction_to_band(comm::CommunicatorGrid& grid, const SizeType b, MatrixRef<T, device>& mat_c,
                          Matrix<const T, device>& mat_v, Matrix<const T, Device::CPU>& mat_taus) {
  DLAF_ASSERT(matrix::equal_process_grid(mat_c, grid), mat_c, grid);
  DLAF_ASSERT(matrix::equal_process_grid(mat_v, grid), mat_v, grid);
  DLAF_ASSERT(square_size(mat_v), mat_v);
  DLAF_ASSERT(square_blocksize(mat_v), mat_v);
  DLAF_ASSERT(mat_c.size().rows() == mat_v.size().rows(), mat_c, mat_v);
  DLAF_ASSERT(mat_c.blockSize().rows() == mat_v.blockSize().rows(), mat_c, mat_v);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_c), mat_c);
  DLAF_ASSERT(matrix::single_tile_per_block(mat_v), mat_v);

  [[maybe_unused]] auto nr_reflectors_blocks = [&b, &mat_v]() {
    const SizeType m = mat_v.size().rows();
    const SizeType mb = mat_v.blockSize().rows();
    return mat_v.distribution().template nextLocalTileFromGlobalTile<Coord::Col>(
        std::max<SizeType>(0, util::ceilDiv(m - b - 1, mb)));
  };
  DLAF_ASSERT(mat_taus.distribution().localNrTiles().rows() == nr_reflectors_blocks(), mat_taus, mat_v,
              b);

  BackTransformationReductionToBand<backend, device, T>::call(grid, b, mat_c, mat_v, mat_taus);
}
}
