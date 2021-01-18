//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

template <class T>
void gemm(const blas::Op op_a, const blas::Op op_b, const T alpha, const Tile<const T, Device::CPU>& a,
          const Tile<const T, Device::CPU>& b, const T beta, const Tile<T, Device::CPU>& c) noexcept {
  auto s = tile::internal::getGemmSizes(op_a, op_b, a, b, c);
  blas::gemm(blas::Layout::ColMajor, op_a, op_b, s.m, s.n, s.k, alpha, a.ptr(), a.ld(), b.ptr(), b.ld(),
             beta, c.ptr(), c.ld());
}

template <class T>
void hemm(const blas::Side side, const blas::Uplo uplo, const T alpha,
          const Tile<const T, Device::CPU>& a, const Tile<const T, Device::CPU>& b, const T beta,
          const Tile<T, Device::CPU>& c) {
  auto s = tile::internal::getHemmSizes(side, a, b, c);
  blas::hemm(blas::Layout::ColMajor, side, uplo, s.m, s.n, alpha, a.ptr(), a.ld(), b.ptr(), b.ld(), beta,
             c.ptr(), c.ld());
}

template <class T>
void her2k(const blas::Uplo uplo, const blas::Op op, const T alpha, const Tile<const T, Device::CPU>& a,
           const Tile<const T, Device::CPU>& b, const BaseType<T> beta, const Tile<T, Device::CPU>& c) {
  auto s = tile::internal::getHer2kSizes(op, a, b, c);
  blas::her2k(blas::Layout::ColMajor, uplo, op, s.n, s.k, alpha, a.ptr(), a.ld(), b.ptr(), b.ld(), beta,
              c.ptr(), c.ld());
}

template <class T>
void herk(const blas::Uplo uplo, const blas::Op op, const BaseType<T> alpha,
          const Tile<const T, Device::CPU>& a, const BaseType<T> beta,
          const Tile<T, Device::CPU>& c) noexcept {
  auto s = tile::internal::getHerkSizes(op, a, c);
  blas::herk(blas::Layout::ColMajor, uplo, op, s.n, s.k, alpha, a.ptr(), a.ld(), beta, c.ptr(), c.ld());
}

template <class T>
void trsm(const blas::Side side, const blas::Uplo uplo, const blas::Op op, const blas::Diag diag,
          const T alpha, const Tile<const T, Device::CPU>& a, const Tile<T, Device::CPU>& b) noexcept {
  auto s = tile::internal::getTrsmSizes(side, a, b);
  blas::trsm(blas::Layout::ColMajor, side, uplo, op, diag, s.m, s.n, alpha, a.ptr(), a.ld(), b.ptr(),
             b.ld());
}
