//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

/// @brief Compute the cholesky decomposition of a.
/// Only the upper or lower triangular elements are referenced according to @p uplo.
/// @returns info = 0 on success or info > 0 if the tile is not positive definite.
/// @throw @c std::runtime_error if the tile was not positive definite.
template <class T, Device device>
long long potrfInfo(blas::Uplo uplo, const Tile<T, device>& a) {
  if (a.size().rows() != a.size().cols()) {
    throw std::invalid_argument("Error: POTRF: A is not square.");
  }

  auto info = lapack::potrf(uplo, a.size().rows(), a.ptr(), a.ld());
  assert(info >= 0);

  return info;
}

/// @brief Compute the cholesky decomposition of a.
/// Only the upper or lower triangular elements are referenced according to @p uplo.
/// @throw @c std::invalid_argument if a is not square.
/// @throw @c std::runtime_error if the tile was not positive definite.
template <class T, Device device>
void potrf(blas::Uplo uplo, const Tile<T, device>& a) {
  auto info = potrfInfo(uplo, a);
  if (info != 0)
    throw std::runtime_error("Error: POTRF: A is not positive definite.");
}
