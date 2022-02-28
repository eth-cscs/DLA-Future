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

#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/util_matrix.h"

#include "dlaf/eigensolver/reduction_to_band/impl.h"

namespace dlaf {
namespace eigensolver {

/// Reduce a local lower Hermitian matrix to symmetric band-diagonal form, with `band = blocksize + 1`.
///
/// See the related distributed version for more details.
//
/// @param mat_a on entry it contains an Hermitian matrix, on exit it is overwritten with the
/// band-diagonal result together with the elementary reflectors. Just the tiles of the lower
/// triangular part will be used.
///
/// @pre mat_a has a square size
/// @pre mat_a has a square block size
/// @pre mat_a is a local matrix
template <Backend backend, Device device, class T>
common::internal::vector<pika::shared_future<common::internal::vector<T>>> reductionToBand(
    Matrix<T, device>& mat_a, const SizeType band_size) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);

  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);

  DLAF_ASSERT(mat_a.blockSize().rows() % band_size == 0, mat_a.blockSize().rows(), band_size);

  return internal::ReductionToBand<backend, device, T>::call(mat_a, band_size);
}

/// Reduce a distributed lower Hermitian matrix to symmetric band-diagonal form, with `band = blocksize + 1`.
///
/// The reduction from a lower Hermitian matrix to the band-diagonal form is performed by an orthogonal
/// similarity transformation Q, applied from left and right as in equation `Q**H . A . Q`, and whose
/// result is stored in-place in @p mat_a.
///
/// The Q matrix is a product of elementary Householder reflectors
/// `Q` = H(1) . H(2) . ... . H(n)
///
/// with `H(i) = I - tau(i) * v(i) . v(i)**H`
///
/// which are stored, together with the resulting band-diagonal matrix, in-place in the lower triangular
/// part of @p mat_a.
///
/// In particular, @p mat_a will look like this (tile representation)
///
/// B ~ ~ ~ ~ ~
/// * B ~ ~ ~ ~
/// v * B ~ ~ ~
/// v v * B ~ ~
/// v v v * B ~
/// v v v v * B
///
/// where each column of `v` is an elementary reflector without its first element (which is always equal
/// to 1), `B` are the tiles containg the band-diagonal form, while `*` tiles contain both elements
/// of the band (upper triangular diagonal included) and of the elementary reflectors (lower triangular
/// diagonal excluded).
///
/// @param grid is the CommunicatorGrid on which @p mat_a is distributed
/// @param mat_a on entry it contains an Hermitian matrix, on exit it is overwritten with the
/// band-diagonal result together with the elementary reflectors as described above. Just the tiles of
/// the lower triangular part will be used.
/// @pre mat_a has a square size
/// @pre mat_a has a square block size
/// @pre mat_a is distributed according to @p grid
template <Backend backend, Device device, class T>
common::internal::vector<pika::shared_future<common::internal::vector<T>>> reductionToBand(
    comm::CommunicatorGrid grid, Matrix<T, device>& mat_a) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);

  return internal::ReductionToBand<backend, device, T>::call(grid, mat_a);
}

/// ---- ETI
#define DLAF_EIGENSOLVER_REDUCTION_TO_BAND_LOCAL_ETI(KWORD, BACKEND, DEVICE, DATATYPE)             \
  KWORD template common::internal::vector<pika::shared_future<common::internal::vector<DATATYPE>>> \
  reductionToBand<BACKEND, DEVICE, DATATYPE>(Matrix<DATATYPE, DEVICE> & mat_a,                     \
                                             const SizeType band_size);

#define DLAF_EIGENSOLVER_REDUCTION_TO_BAND_DISTR_ETI(KWORD, BACKEND, DEVICE, DATATYPE)             \
  KWORD template common::internal::vector<pika::shared_future<common::internal::vector<DATATYPE>>> \
  reductionToBand<BACKEND, DEVICE, DATATYPE>(comm::CommunicatorGrid grid,                          \
                                             Matrix<DATATYPE, DEVICE> & mat_a);

#define DLAF_EIGENSOLVER_REDUCTION_TO_BAND_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  DLAF_EIGENSOLVER_REDUCTION_TO_BAND_LOCAL_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  DLAF_EIGENSOLVER_REDUCTION_TO_BAND_DISTR_ETI(KWORD, BACKEND, DEVICE, DATATYPE)

DLAF_EIGENSOLVER_REDUCTION_TO_BAND_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_EIGENSOLVER_REDUCTION_TO_BAND_ETI(extern, Backend::MC, Device::CPU, double)
DLAF_EIGENSOLVER_REDUCTION_TO_BAND_ETI(extern, Backend::MC, Device::CPU, std::complex<float>)
DLAF_EIGENSOLVER_REDUCTION_TO_BAND_ETI(extern, Backend::MC, Device::CPU, std::complex<double>)

#ifdef DLAF_WITH_CUDA
DLAF_EIGENSOLVER_REDUCTION_TO_BAND_LOCAL_ETI(extern, Backend::GPU, Device::GPU, float)
DLAF_EIGENSOLVER_REDUCTION_TO_BAND_LOCAL_ETI(extern, Backend::GPU, Device::GPU, double)
DLAF_EIGENSOLVER_REDUCTION_TO_BAND_LOCAL_ETI(extern, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_EIGENSOLVER_REDUCTION_TO_BAND_LOCAL_ETI(extern, Backend::GPU, Device::GPU, std::complex<double>)
#endif
}
}
