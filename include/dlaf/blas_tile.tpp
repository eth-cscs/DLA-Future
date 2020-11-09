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

  DLAF_ASSERT(m == c.size().rows(), "`m` cannot be determined!", m, c);
  DLAF_ASSERT(n == c.size().cols(), "`n` cannot be determined!", n, c);
  DLAF_ASSERT(k == k2, "`k` cannot be determined!", k, k2);

  blas::gemm(blas::Layout::ColMajor, op_a, op_b, m, n, k, alpha, a.ptr(), a.ld(), b.ptr(), b.ld(), beta,
             c.ptr(), c.ld());
}

template <class T, Device device>
void hemm(const blas::Side side, const blas::Uplo uplo, const T alpha, const Tile<const T, device>& a,
          const Tile<const T, device>& b, const T beta, const Tile<T, device>& c) {
  const SizeType m = c.size().rows();
  const SizeType n = c.size().cols();

  if (side == blas::Side::Left) {
    DLAF_ASSERT(m == a.size().rows(), "`m` cannot be determined!", m, a);
    DLAF_ASSERT(n == b.size().cols(), "`n` cannot be determined!", n, b);
    DLAF_ASSERT(a.size().cols() == b.size().rows(),
                "columns of matrix `a` does not correspond to rows of matrix `b`", a, b);
  }
  else if (side == blas::Side::Right) {
    DLAF_ASSERT(m == b.size().rows(), "`m` cannot be determined!", m, b);
    DLAF_ASSERT(n == a.size().cols(), "`n` cannot be determined!", n, a);
    DLAF_ASSERT(a.size().rows() == b.size().cols(),
                "rows of matrix `a` does not correspond to columns of matrix `b`", a, b);
  }

  blas::hemm(blas::Layout::ColMajor, side, uplo, m, n, alpha, a.ptr(), a.ld(), b.ptr(), b.ld(), beta,
             c.ptr(), c.ld());
}

template <class T, Device device>
void her2k(const blas::Uplo uplo, const blas::Op op, const T alpha, const Tile<const T, device>& a,
           const Tile<const T, device>& b, const BaseType<T> beta, const Tile<T, device>& c) {
  const SizeType n = (op == blas::Op::NoTrans) ? a.size().rows() : a.size().cols();
  const SizeType k = (op == blas::Op::NoTrans) ? a.size().cols() : a.size().rows();

  DLAF_ASSERT((!std::is_same<T, ComplexType<T>>::value || op != blas::Op::Trans),
              "`op = Trans` is not allowed in complex HER2K.");
  DLAF_ASSERT(c.size().rows() == c.size().cols(), "`c` is not square!", c);
  DLAF_ASSERT(c.size().rows() == n, "`c` has an invalid size!", c, n);

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

  DLAF_ASSERT((!std::is_same<T, ComplexType<T>>::value || op != blas::Op::Trans),
              "op = Trans is not allowed for Complex values!");
  DLAF_ASSERT(c.size().rows() == c.size().cols(), "`c` is not square!", c);
  DLAF_ASSERT(c.size().rows() == n, "`c` has an invalid size!", c, n);

  blas::herk(blas::Layout::ColMajor, uplo, op, n, k, alpha, a.ptr(), a.ld(), beta, c.ptr(), c.ld());
}

template <class T, Device device>
void trsm(const blas::Side side, const blas::Uplo uplo, const blas::Op op, const blas::Diag diag,
          const T alpha, const Tile<const T, device>& a, const Tile<T, device>& b) noexcept {
  const SizeType m = b.size().rows();
  const SizeType n = b.size().cols();

  DLAF_ASSERT(a.size().rows() == a.size().cols(), "`a` is not square!", a);

  const auto left_side = (side == blas::Side::Left ? m : n);
  DLAF_ASSERT(a.size().rows() == left_side, "`a` has an invalid size!", a, left_side);

  blas::trsm(blas::Layout::ColMajor, side, uplo, op, diag, m, n, alpha, a.ptr(), a.ld(), b.ptr(),
             b.ld());
}
