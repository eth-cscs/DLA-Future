//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <functional>
#include <tuple>

#include <blas.hh>

#include <dlaf_test/util_types.h>

/// @file

namespace dlaf {
namespace matrix {
namespace test {

/// Returns a tuple of element generators of two matrices A(n x n) and T(n x n).
/// such that A = T T^H if @p uplo == Lower
/// or        A = T^H T if @p uplo == Upper
///
/// The tile elements are chosen such that:
/// t_ij = 1 / 2^(|i-j|) * exp(I*(-i+j)),
/// where I = 0 for real types or I is the complex unit for complex types.
/// By definition A should be:
/// a_ij = Sum_k(res_ik * ConjTrans(res)_kj) =
///      = Sum_k(1 / 2^(|i-k| + |j-k|) * exp(I*(-i+j))),
/// where k = 0 .. min(i,j)
/// Therefore,
/// a_ij = (4^(min(i,j)+1) - 1) / (3 * 2^(i+j)) * exp(I*(-i+j))
template <class ElementIndex, class T>
auto getCholeskySetters(blas::Uplo uplo) {
  using dlaf::test::TypeUtilities;

  std::function<T(const ElementIndex&)> el_a = [uplo](const ElementIndex& index) {
    if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
        (uplo == blas::Uplo::Upper && index.row() > index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    const double i = index.row();
    const double j = index.col();

    return TypeUtilities<T>::polar(std::exp2(-(i + j)) / 3 * (std::exp2(2 * (std::min(i, j) + 1)) - 1),
                                   -i + j);
  };

  // Analytical results
  std::function<T(const ElementIndex&)> el_t = [uplo](const ElementIndex& index) {
    if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
        (uplo == blas::Uplo::Upper && index.row() > index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    const double i = index.row();
    const double j = index.col();

    return TypeUtilities<T>::polar(std::exp2(-std::abs(i - j)), -i + j);
  };

  return std::make_tuple(el_a, el_t);
}

/// Returns a tuple of element generators of three matrices T (n x n), A(n x n) and B (n x n).
/// It holds, for @p itype == 1
/// B = U^(-H) A U^(-1), if @p uplo == Upper,
/// B = L^(-1) A L^(-H), if @p uplo == Lower;
/// while for @p itype = 2 and @p itype == 3
/// B = U A U^(H), if @p uplo == Upper,
/// B = L^(H) A L, if @p uplo == Lower.
///
/// The elements of T (@p el_T), a triangular matrix, are computed, for @p itype == 1, as
/// T_ij = beta / 2^(i-j) exp(I alpha (i-j))
/// and, for @p itype == 2 or @p itype == 3, as
/// T_ij = beta / 2^(j-i) exp(I alpha (i-j))
/// where alpha = 0 and I = 0 for real types or I is the complex unit for complex types.
///
/// The elements of the Hermitian matrix A (@p el_a) are chosen such that for @p itype == 1:
/// A_ij = ((i+1)(j+1)(beta^2 gamma)) / (2^(i+j)) exp(I alpha (i-j))
/// or for @p itype == 2 and @p itype == 3:
/// A_ij = gamma / 2^(i+j) exp(I alpha (i-j))
///
/// Finally, the elements of B (@p el_b) should be, for @p itype == 1:
/// B_ij = gamma / 2^(i+j) exp(I alpha (i-j))
/// and for @p itype == 2 or @p itype == 3:
/// B_ij = ((n-i) (n-j) (beta^2 gamma)) / 2^(i+j) exp(I alpha (i-j))
/// where @p n is the size of the matrix.
///
template <class ElementIndex, class T>
auto getGenToStdElementSetters(SizeType n, int itype, blas::Uplo uplo, BaseType<T> alpha,
                               BaseType<T> beta, BaseType<T> gamma) {
  using dlaf::test::TypeUtilities;

  DLAF_ASSERT(uplo == blas::Uplo::Lower || uplo == blas::Uplo::Upper, uplo);
  DLAF_ASSERT(itype >= 1 && itype <= 3, itype);
  DLAF_ASSERT(gamma > 0, gamma);

  std::function<T(const ElementIndex&)> el_t = [itype, uplo, alpha, beta](const ElementIndex& index) {
    if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
        (uplo == blas::Uplo::Upper && index.row() > index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    const double i = index.row();
    const double j = index.col();
    if (itype == 1)
      return TypeUtilities<T>::polar(beta / std::exp2(std::abs(i - j)), alpha * (i - j));
    else
      return TypeUtilities<T>::polar(beta * std::exp2(std::abs(i - j)), alpha * (i - j));
  };

  std::function<T(const ElementIndex&)> el_a = [itype, uplo, alpha, beta,
                                                gamma](const ElementIndex& index) {
    if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
        (uplo == blas::Uplo::Upper && index.row() > index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    const double i = index.row();
    const double j = index.col();
    if (itype == 1)
      return TypeUtilities<T>::polar((i + 1) * (j + 1) * (beta * beta * gamma) / std::exp2(i + j),
                                     alpha * (i - j));
    else
      return TypeUtilities<T>::polar(gamma / std::exp2(i + j), alpha * (i - j));
  };

  std::function<T(const ElementIndex&)> el_b = [n, itype, uplo, alpha, beta,
                                                gamma](const ElementIndex& index) {
    if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
        (uplo == blas::Uplo::Upper && index.row() > index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    const double i = index.row();
    const double j = index.col();

    if (itype == 1)
      return TypeUtilities<T>::polar(gamma / std::exp2(i + j), alpha * (i - j));
    else
      return TypeUtilities<T>::polar((n - i) * (n - j) * (beta * beta * gamma) / std::exp2(i + j),
                                     alpha * (i - j));
  };

  return std::make_tuple(el_t, el_a, el_b);
}

/// Returns a tuple of element generators of two matrices A(n x n) and T(n x n).
/// such that A = T^H T if @p uplo == Lower
/// or        A = T T^H if @p uplo == Upper
///
/// The tile elements are chosen such that:
/// t_ij = 1 / 2^(|i-j|) * exp(I*(-i+j)),
/// where I = 0 for real types or I is the complex unit for complex types.
/// By definition A should be:
/// a_lh = Sum_k(res_ik * ConjTrans(res)_kj) =
///      = Sum_k(1 / 2^(|i-k| + |j-k|) * exp(I*(-i+j))),
/// where k = 0 .. min(i,j), l = n-1-i, h = n-1-j
/// Therefore,
/// a_lh = (4^(min(i,j)+1) - 1) / (3 * 2^(i+j)) * exp(I*(i-j))
template <class ElementIndex, class T>
auto get_assemble_cholesky_inverse_setters(SizeType n, blas::Uplo uplo) {
  using dlaf::test::TypeUtilities;

  std::function<T(const ElementIndex&)> el_t = [uplo](const ElementIndex& index) {
    if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
        (uplo == blas::Uplo::Upper && index.row() > index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    const double i = index.row();
    const double j = index.col();

    return TypeUtilities<T>::polar(std::exp2(-std::abs(i - j)), -i + j);
  };

  // Analytical results
  std::function<T(const ElementIndex&)> el_a = [n, uplo](const ElementIndex& index) {
    if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
        (uplo == blas::Uplo::Upper && index.row() > index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    const double i = n - 1 - index.row();
    const double j = n - 1 - index.col();

    return TypeUtilities<T>::polar(std::exp2(-(i + j)) / 3 * (std::exp2(2 * (std::min(i, j) + 1)) - 1),
                                   i - j);
  };

  return std::make_tuple(el_t, el_a);
}

/// Returns a tuple of element generators of two matrices A(n x n) and T(n x n).
/// such that A = inv(T^H) inv(T) if @p uplo == Lower
/// or        A = inv(T) inv(T^H) if @p uplo == Upper
///
/// The tile elements are chosen such that:
/// t_ij = 1                    in the diagonal
/// t_ij = -0.5 * exp(I*(-i+j)) in the offdiagonal,
/// where I = 0 for real types or I is the complex unit for complex types.
/// i.e.
/// inv(t)_ij = 1 / 2^(|i-j|) * exp(I*(-i+j)),
/// By definition A should be:
/// a_lh = Sum_k(res_ik * ConjTrans(res)_kj) =
///      = Sum_k(1 / 2^(|i-k| + |j-k|) * exp(I*(-i+j))),
/// where k = 0 .. min(i,j), l = n-1-i, h = n-1-j
/// Therefore,
/// a_lh = (4^(min(i,j)+1) - 1) / (3 * 2^(i+j)) * exp(I*(i-j))
template <class ElementIndex, class T>
auto get_inverse_cholesky_factor_setters(SizeType n, blas::Uplo uplo) {
  using dlaf::test::TypeUtilities;

  std::function<T(const ElementIndex&)> el_t = [uplo](const ElementIndex& index) {
    if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
        (uplo == blas::Uplo::Upper && index.row() > index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    const double i = index.row();
    const double j = index.col();

    if (i == j)
      return TypeUtilities<T>::polar(1, 0);
    if (i == j + 1 || i == j - 1)
      return -TypeUtilities<T>::polar(0.5, -i + j);
    return TypeUtilities<T>::polar(0, 0);
  };

  // Analytical results
  std::function<T(const ElementIndex&)> el_a = [n, uplo](const ElementIndex& index) {
    if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
        (uplo == blas::Uplo::Upper && index.row() > index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    const double i = n - 1 - index.row();
    const double j = n - 1 - index.col();

    return TypeUtilities<T>::polar(std::exp2(-(i + j)) / 3 * (std::exp2(2 * (std::min(i, j) + 1)) - 1),
                                   i - j);
  };

  return std::make_tuple(el_t, el_a);
}

/// Returns a tuple of element generators of two triangular matrices A(n x n) and B(n x n).
/// such that A = inv(B)
///
/// The tile elements are chosen such that:
/// diag == Unit (i.e. diagonal element are not set but read 1)
/// a_ij = 1 / 2^(|i-j|) * exp(I*(-i+j)),
/// b_ij = 1                    in the diagonal
/// b_ij = -0.5 * exp(I*(-i+j)) in the offdiagonal,
/// where I = 0 for real types or I is the complex unit for complex types.
/// diag == NonUnit
/// a_ij = 4 / 2^(|i-j|) * exp(I*(-i+j)),
/// b_ij = 1 / 4                  in the diagonal
/// b_ij = -1 / 8 * exp(I*(-i+j)) in the offdiagonal,
/// where I = 0 for real types or I is the complex unit for complex types.
template <class ElementIndex, class T>
auto get_triangular_inverse_setters(blas::Uplo uplo, blas::Diag diag) {
  using dlaf::test::TypeUtilities;

  if (diag == blas::Diag::Unit) {
    std::function<T(const ElementIndex&)> el_a = [uplo](const ElementIndex& index) {
      if ((uplo == blas::Uplo::Lower && index.row() <= index.col()) ||
          (uplo == blas::Uplo::Upper && index.row() >= index.col()))
        return TypeUtilities<T>::element(-9.9, 0);

      const double i = index.row();
      const double j = index.col();

      return TypeUtilities<T>::polar(std::exp2(-std::abs(i - j)), -i + j);
    };

    // Analytical results
    std::function<T(const ElementIndex&)> res_b = [uplo](const ElementIndex& index) {
      if ((uplo == blas::Uplo::Lower && index.row() <= index.col()) ||
          (uplo == blas::Uplo::Upper && index.row() >= index.col()))
        return TypeUtilities<T>::element(-9.9, 0);

      const double i = index.row();
      const double j = index.col();

      if (i == j + 1 || i == j - 1)
        return -TypeUtilities<T>::polar(0.5, -i + j);
      return TypeUtilities<T>::polar(0, 0);
    };

    return std::make_tuple(el_a, res_b);
  }
  else {
    std::function<T(const ElementIndex&)> el_a = [uplo](const ElementIndex& index) {
      if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
          (uplo == blas::Uplo::Upper && index.row() > index.col()))
        return TypeUtilities<T>::element(-9.9, 0);

      const double i = index.row();
      const double j = index.col();

      return TypeUtilities<T>::polar(4 * std::exp2(-std::abs(i - j)), -i + j);
    };

    // Analytical results
    std::function<T(const ElementIndex&)> res_b = [uplo](const ElementIndex& index) {
      if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
          (uplo == blas::Uplo::Upper && index.row() > index.col()))
        return TypeUtilities<T>::element(-9.9, 0);

      const double i = index.row();
      const double j = index.col();

      if (i == j)
        return TypeUtilities<T>::polar(0.25, 0);
      if (i == j + 1 || i == j - 1)
        return -TypeUtilities<T>::polar(0.125, -i + j);
      return TypeUtilities<T>::polar(0, 0);
    };

    return std::make_tuple(el_a, res_b);
  }
}

}
}
}
