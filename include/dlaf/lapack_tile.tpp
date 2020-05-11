//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

template <class T, Device device>
dlaf::BaseType<T> lange(lapack::Norm norm, const Tile<T, device>& a) {
  return lapack::lange(norm, a.size().cols(), a.size().rows(), a.ptr(), a.ld());
}

template <class T, Device device>
dlaf::BaseType<T> lantr(lapack::Norm norm, blas::Uplo uplo, blas::Diag diag, const Tile<T, device>& a) {
  return lapack::lantr(norm, uplo, diag, a.size().cols(), a.size().rows(), a.ptr(), a.ld());
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
