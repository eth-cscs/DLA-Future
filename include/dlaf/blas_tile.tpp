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
void gemm(blas::Op op_a, blas::Op op_b, T alpha, const Tile<const T, device>& a,
          const Tile<const T, device>& b, T beta, const Tile<T, device>& c) {
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

  DLAF_ASSERT((m == c.size().rows()), "GEMM: m cannot be determined.");
  DLAF_ASSERT((n == c.size().cols()), "GEMM: n cannot be determined.");
  DLAF_ASSERT((k == k2), "GEMM: k cannot be determined.");

  blas::gemm(blas::Layout::ColMajor, op_a, op_b, m, n, k, alpha, a.ptr(), a.ld(), b.ptr(), b.ld(), beta,
             c.ptr(), c.ld());
}

template <class T, Device device>
void herk(blas::Uplo uplo, blas::Op op, BaseType<T> alpha, const Tile<const T, device>& a,
          BaseType<T> beta, const Tile<T, device>& c) {
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
              "Complex HERK: op = Trans is not allowed.");
  DLAF_ASSERT((c.size().rows() == c.size().cols()), "HERK: C is not square.");
  DLAF_ASSERT((c.size().rows() == n), "HERK: C has an invalid size.");

  blas::herk(blas::Layout::ColMajor, uplo, op, n, k, alpha, a.ptr(), a.ld(), beta, c.ptr(), c.ld());
}

template <class T, Device device>
void trsm(blas::Side side, blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha,
          const Tile<const T, device>& a, const Tile<T, device>& b) {
  SizeType m = b.size().rows();
  SizeType n = b.size().cols();

  DLAF_ASSERT((a.size().rows() == a.size().cols()), "TRSM: A is not square.");
  DLAF_ASSERT((a.size().rows() == (side == blas::Side::Left ? m : n)), "TRSM: A has an invalid size.");

  blas::trsm(blas::Layout::ColMajor, side, uplo, op, diag, m, n, alpha, a.ptr(), a.ld(), b.ptr(),
             b.ld());
}
