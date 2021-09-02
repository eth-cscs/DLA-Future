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
#include "dlaf/common/assert.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/eigensolver/band_to_tridiag/mc.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {

/// Reduces a Hermitian band matrix A (with the given band_size(*)) to real symmetric tridiagonal
/// form T by a unitary similarity transformation Q**H * A * Q = T.
///
/// Q is represented by a series of HouseHolder transformations.
/// The tridiagonal matrix returned is stored in a local Matrix, where:
/// - row 0 contains the diagonal
/// - row 1 contains the off-diagonal.
/// As the offdiagonal is shorter, the last element of row 1 is not used.
/// The HH Reflectors are returned in a compact form.
/// As the first non-zero element of the vectors is always 1 it is replaced by the corresponding tau.
/// The matrix Q is computed in the following way:
/// Real Type:
/// HHT(0, 0) HHT(0, 1) .. HHT(0, last(0)) HHT(1, 0) .. HHT(1, last(1)) .. HHT(m - 3, 0).
/// Complex Type:
/// HHT(0, 0) HHT(0, 1) .. HHT(0, last(0)) HHT(1, 0) .. HHT(1, last(1)) .. HHT(m - 3, 0) HHT(m - 2, 0)
/// where m is the size of A and HHT(sw, st) is the HH transformation given by (Id - v tau v^H),
/// last(sw) = ceilDiv(m - sw - 2, band_size), (in case of complex types last(m - 2) = 1)
/// and tau and v are defined in the following way:
/// - start = 1 + sw + st * band_size
/// - len = min(band_size, m - start)
/// - pos = (sw / band_size + st) * band_size
/// - v[start] = 1
/// - tau = V(pos, sw) (Note: integer division)
/// - v[start + 1:start + len] = V(pos + 1 : pos + len, sw)
/// - all other elements of v are 0
///
/// E.g. for m = 6 (left) and m = 7 (right) (in both cases band_size = 3)
/// V haves the following layouts:
/// / A0 B0 C0 ** ** ** \                    / A0 B0 C0 ** ** ** ** \
/// | A0 B0 C0 ** ** ** |                    | A0 B0 C0 ** ** ** ** |
/// | A0 B0 C0 ** ** ** |                    | A0 B0 C0 ** ** ** ** |
/// | A1 ** ** D0 E0 ** |                    | A1 B1 ** D0 E0 F0 ** |
/// | A1 ** ** D0 ** ** |                    | A1 B1 ** D0 E0 ** ** |
/// \ ** ** ** ** ** ** /                    | A1 ** ** D0 ** ** ** |
/// (E0 only fom complex types)              \ ** ** ** ** ** ** ** /
///                                         (F0 only fom complex types)
///
/// Note the letter denotes sw (A=0, B=1, ...), and the number st.
/// (*) diagonal + band_size off-diagonals.
///
/// Implementation on local memory.
///
/// @param mat_a contains the Hermitian band matrix A (if A is real, the matrix is symmetric).
/// @pre mat_a has a square size,
/// @pre mat_a has a square block size,
/// @pre band_size is a divisor of mat_a.blockSize().cols(),
/// @pre mat_a is not distributed.
template <Backend backend, Device device, class T>
ReturnTridiagType<T, device> bandToTridiag(blas::Uplo uplo, SizeType band_size,
                                           Matrix<const T, device>& mat_a) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  DLAF_ASSERT(mat_a.blockSize().rows() % band_size == 0, mat_a.blockSize().rows(), band_size);
  DLAF_ASSERT(matrix::local_matrix(mat_a), mat_a);

  switch (uplo) {
    case blas::Uplo::Lower:
      return internal::BandToTridiag<backend, device, T>::call_L(band_size, mat_a);
      break;
    case blas::Uplo::Upper:
      DLAF_UNIMPLEMENTED(uplo);
      break;
    case blas::Uplo::General:
      DLAF_UNIMPLEMENTED(uplo);
      break;
  }

  return DLAF_UNREACHABLE(ReturnTridiagType<T, device>);
}

}
}
