//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/common/assert.h"

template <class T>
void lacpy(const Tile<const T, Device::CPU>& a, const Tile<T, Device::CPU>& b) {
  DLAF_ASSERT_MODERATE(a.size() == b.size(),
                       "Source and destination tile must have the same size. A=", a.size(),
                       " B=", b.size());

  SizeType m = a.size().rows();
  SizeType n = a.size().cols();

  lapack::lacpy(lapack::MatrixType::General, m, n, a.ptr(), a.ld(), b.ptr(), b.ld());
}

template <class T, Device device>
dlaf::BaseType<T> lange(lapack::Norm norm, const Tile<T, device>& a) noexcept {
  return lapack::lange(norm, a.size().rows(), a.size().cols(), a.ptr(), a.ld());
}

template <class T, Device device>
dlaf::BaseType<T> lantr(lapack::Norm norm, blas::Uplo uplo, blas::Diag diag,
                        const Tile<T, device>& a) noexcept {
  switch (uplo) {
    case blas::Uplo::Lower:
      DLAF_ASSERT(a.size().rows() >= a.size().cols(), a);
      break;
    case blas::Uplo::Upper:
      DLAF_ASSERT(a.size().rows() <= a.size().cols(), a);
      break;
    case blas::Uplo::General:
      DLAF_ASSERT(blas::Uplo::General == uplo, "Invalid parameter");
      break;
  }
  return lapack::lantr(norm, uplo, diag, a.size().rows(), a.size().cols(), a.ptr(), a.ld());
}

template <class T, Device device>
void potrf(blas::Uplo uplo, const Tile<T, device>& a) {
  auto info = potrfInfo(uplo, a);
  if (info != 0)
    throw std::runtime_error("Error: POTRF: A is not positive definite.");
}

template <class T, Device device>
long long potrfInfo(blas::Uplo uplo, const Tile<T, device>& a) {
  if (a.size().rows() != a.size().cols()) {
    throw std::invalid_argument("Error: POTRF: A is not square.");
  }

  auto info = lapack::potrf(uplo, a.size().rows(), a.ptr(), a.ld());
  assert(info >= 0);

  return info;
}
