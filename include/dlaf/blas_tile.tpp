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
void gemm(const blas::Op op_a, const blas::Op op_b, const T alpha, const Tile<const T, device>& a,
          const Tile<const T, device>& b, const T beta, const Tile<T, device>& c) noexcept {
  SizeType m;
  SizeType k;
  if (op_a == blas::Op::NoTrans) {
    m = a.size().rows();
    k = a.size().cols();
  }
  else {
    m = a.size().cols();
    k = a.size().rows();
  }
  SizeType k2;
  SizeType n;
  if (op_b == blas::Op::NoTrans) {
    k2 = b.size().rows();
    n = b.size().cols();
  }
  else {
    k2 = b.size().cols();
    n = b.size().rows();
  }

  DLAF_ASSERT(m == c.size().rows(), op_a, a, c);
  DLAF_ASSERT(n == c.size().cols(), op_b, b, c);
  DLAF_ASSERT(k == k2, op_a, a, op_b, b);

  blas::gemm(blas::Layout::ColMajor, op_a, op_b, m, n, k, alpha, a.ptr(), a.ld(), b.ptr(), b.ld(), beta,
             c.ptr(), c.ld());
}

template <class T, Device device>
void hemm(const blas::Side side, const blas::Uplo uplo, const T alpha, const Tile<const T, device>& a,
          const Tile<const T, device>& b, const T beta, const Tile<T, device>& c) {
  const SizeType m = c.size().rows();
  const SizeType n = c.size().cols();

  if (side == blas::Side::Left) {
    DLAF_ASSERT(m == a.size().rows(), c, a);
    DLAF_ASSERT(n == b.size().cols(), c, b);
    DLAF_ASSERT(a.size().cols() == b.size().rows(), a, b);
  }
  else if (side == blas::Side::Right) {
    DLAF_ASSERT(m == b.size().rows(), c, b);
    DLAF_ASSERT(n == a.size().cols(), c, a);
    DLAF_ASSERT(a.size().rows() == b.size().cols(), a, b);
  }

  blas::hemm(blas::Layout::ColMajor, side, uplo, m, n, alpha, a.ptr(), a.ld(), b.ptr(), b.ld(), beta,
             c.ptr(), c.ld());
}

template <class T, Device device>
void her2k(const blas::Uplo uplo, const blas::Op op, const T alpha, const Tile<const T, device>& a,
           const Tile<const T, device>& b, const BaseType<T> beta, const Tile<T, device>& c) {
  const SizeType n = (op == blas::Op::NoTrans) ? a.size().rows() : a.size().cols();
  const SizeType k = (op == blas::Op::NoTrans) ? a.size().cols() : a.size().rows();

  DLAF_ASSERT(tile_complex_trans<T>(op), op);

  DLAF_ASSERT(square_size(c), c);
  DLAF_ASSERT(c.size().rows() == n, c, op, a);

  blas::her2k(blas::Layout::ColMajor, uplo, op, n, k, alpha, a.ptr(), a.ld(), b.ptr(), b.ld(), beta,
              c.ptr(), c.ld());
}

template <class T, Device device>
void herk(const blas::Uplo uplo, const blas::Op op, const BaseType<T> alpha,
          const Tile<const T, device>& a, const BaseType<T> beta, const Tile<T, device>& c) noexcept {
  SizeType n;
  SizeType k;
  if (op == blas::Op::NoTrans) {
    n = a.size().rows();
    k = a.size().cols();
  }
  else {
    n = a.size().cols();
    k = a.size().rows();
  }

  DLAF_ASSERT(tile_complex_trans<T>(op), op);
  DLAF_ASSERT(square_size(c), c);
  DLAF_ASSERT(c.size().rows() == n, c, op, a);

  blas::herk(blas::Layout::ColMajor, uplo, op, n, k, alpha, a.ptr(), a.ld(), beta, c.ptr(), c.ld());
}

template <class T, Device device>
void trsm(const blas::Side side, const blas::Uplo uplo, const blas::Op op, const blas::Diag diag,
          const T alpha, const Tile<const T, device>& a, const Tile<T, device>& b) noexcept {
  const SizeType m = b.size().rows();
  const SizeType n = b.size().cols();

  DLAF_ASSERT(square_size(a), a);

  const auto left_side = (side == blas::Side::Left ? m : n);
  DLAF_ASSERT(a.size().rows() == left_side, a, op, b);

  blas::trsm(blas::Layout::ColMajor, side, uplo, op, diag, m, n, alpha, a.ptr(), a.ld(), b.ptr(),
             b.ld());
}
