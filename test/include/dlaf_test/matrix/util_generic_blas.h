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

#include <functional>
#include <tuple>

#include <blas.hh>

#include "dlaf_test/util_types.h"

/// @file

namespace dlaf {
namespace matrix {
namespace test {

/// Returns a tuple of element generators of three matrices A(m x m), B (m x n), X (m x n), for which it
/// holds op(A) X = alpha B (n can be any value).
///
/// The elements of op(A) (@p el_op_a) are chosen such that:
///   op(A)_ik = (i+1) / (k+.5) * exp(I*(2*i-k)) for the referenced elements,
///   op(A)_ik = -9.9 otherwise,
/// where I = 0 for real types or I is the complex unit for complex types.
///
/// The elements of X (@p el_x) are computed as
///   X_kj = (k+.5) / (j+2) * exp(I*(k+j)).
/// These data are typically used to check whether the result of the equation
/// performed with any algorithm is consistent with the computed values.
///
/// Finally, the elements of B (@p el_b) should be:
/// B_ij = (Sum_k op(A)_ik * X_kj) / alpha
///      = (op(A)_ii * X_ij + (kk-1) * gamma) / alpha,
/// where gamma = (i+1) / (j+2) * exp(I*(2*i+j)),
///       kk = i+1 if op(a) is an lower triangular matrix, or
///       kk = m-i if op(a) is an lower triangular matrix.
/// Therefore
/// B_ij = (X_ij + (kk-1) * gamma) / alpha, if diag == Unit,
/// B_ij = kk * gamma / alpha, otherwise.
///
template <class ElementIndex, class T>
auto getLeftTriangularSystem(blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha, SizeType m) {
  using dlaf::test::TypeUtilities;

  const bool op_a_lower = ((uplo == blas::Uplo::Lower && op == blas::Op::NoTrans) ||
                           (uplo == blas::Uplo::Upper && op != blas::Op::NoTrans));

  std::function<T(const ElementIndex&)> el_op_a = [op_a_lower, diag](const ElementIndex& index) {
    if ((op_a_lower && index.row() < index.col()) || (!op_a_lower && index.row() > index.col()) ||
        (diag == blas::Diag::Unit && index.row() == index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    const double i = index.row();
    const double k = index.col();

    return TypeUtilities<T>::polar((i + 1) / (k + .5), 2 * i - k);
  };

  std::function<T(const ElementIndex&)> el_x = [](const ElementIndex& index) {
    const double k = index.row();
    const double j = index.col();

    return TypeUtilities<T>::polar((k + .5) / (j + 2), k + j);
  };

  std::function<T(const ElementIndex&)> el_b = [m, alpha, diag, op_a_lower,
                                                el_x](const ElementIndex& index) {
    BaseType<T> kk = op_a_lower ? index.row() + 1 : m - index.row();

    const double i = index.row();
    const double j = index.col();
    const T gamma = TypeUtilities<T>::polar((i + 1) / (j + 2), 2 * i + j);
    if (diag == blas::Diag::Unit)
      return ((kk - 1) * gamma + el_x(index)) / alpha;
    else
      return kk * gamma / alpha;
  };

  return std::make_tuple(el_op_a, el_b, el_x);
}

/// Returns a tuple of element generators of three matrices A(m x m), B (m x n), X (m x n), for which it
/// holds X op(A) = alpha B (n can be any value).
///
/// The elements of op(A) (@p el_op_a) are chosen such that:
///   op(A)_kj = (j+1) / (k+.5) * exp(I*(2*j-k)) for the referenced elements,
///   op(A)_kj = -9.9 otherwise,
/// where I = 0 for real types or I is the complex unit for complex types.
///
/// The elements of X (@p el_x) are computed as
///   X_ik = (k+.5) / (i+2) * exp(I*(i+k)).
/// These data are typically used to check whether the result of the equation
/// performed with any algorithm is consistent with the computed values.
///
/// Finally, the elements of B (@p el_b) should be:
/// B_ij = (Sum_k X_ik * op(A)_kj) / alpha
///      = (X_ij * op(A)_jj + (kk-1) * gamma) / alpha,
/// where gamma = (j+1) / (i+2) * exp(I*(i+2*j)),
///       kk = j+1 if op(a) is an upper triangular matrix, or
///       kk = m-j if op(a) is an upper triangular matrix.
/// Therefore
/// B_ij = (X_ij + (kk-1) * gamma) / alpha, if diag == Unit,
/// B_ij = kk * gamma / alpha, otherwise.
///
template <class ElementIndex, class T>
auto getRightTriangularSystem(blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha, SizeType n) {
  using dlaf::test::TypeUtilities;

  const bool op_a_lower = ((uplo == blas::Uplo::Lower && op == blas::Op::NoTrans) ||
                           (uplo == blas::Uplo::Upper && op != blas::Op::NoTrans));

  auto el_x = [](const ElementIndex& index) {
    const double i = index.row();
    const double k = index.col();

    return TypeUtilities<T>::polar((k + .5) / (i + 2), i + k);
  };

  auto el_op_a = [op_a_lower, diag](const ElementIndex& index) {
    if ((op_a_lower && index.row() < index.col()) || (!op_a_lower && index.row() > index.col()) ||
        (diag == blas::Diag::Unit && index.row() == index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    const double k = index.row();
    const double j = index.col();

    return TypeUtilities<T>::polar((j + 1) / (k + .5), 2 * j - k);
  };

  auto el_b = [n, alpha, diag, op_a_lower, el_x](const ElementIndex& index) {
    BaseType<T> kk = op_a_lower ? n - index.col() : index.col() + 1;

    const double i = index.row();
    const double j = index.col();
    const T gamma = TypeUtilities<T>::polar((j + 1) / (i + 2), i + 2 * j);
    if (diag == blas::Diag::Unit)
      return ((kk - 1) * gamma + el_x(index)) / alpha;
    else
      return kk * gamma / alpha;
  };

  return std::make_tuple(el_op_a, el_b, el_x);
}

/// Returns a tuple of element generators of three matrices A(m x m), B (m x n), X (m x n), for which it
/// holds X = alpha op(A) B (n can be any value).
///
/// The elements of op(A) (@p el_op_a) are chosen such that:
///   op(A)_ki = (i+1) / (k+.5) * exp(I*(2*i-k)) for the referenced elements
///   op(A)_ki = -9.9 otherwise,
/// where I = 0 for real types or I is the complex unit for complex types.
///
/// The elements of X (@p el_b) are computed as
///   B_kj = (k+.5) / (j+2) * exp(I*(k+j)).
/// These data are typically used to check whether the result of the equation
/// performed with any algorithm is consistent with the computed values.
///
/// Finally, the elements of X (@p el_x) should be:
/// X_ij = (Sum_k op(A)_ik * B_kj) * alpha
///      = (op(A)_ii * B_ij + (kk-1) * gamma) * alpha,
/// where gamma = (i+1) / (j+2) * exp(I*(2*i+j)),
///       kk = i+1 if op(a) is an lower triangular matrix, or
///       kk = m-i if op(a) is an lower triangular matrix.
/// Therefore
/// X_ij = (B_ij + (kk-1) * gamma) * alpha, if diag == Unit
/// X_ij = kk * gamma * alpha, otherwise.
///
template <class ElementIndex, class T>
auto getLeftTriangularMMSystem(blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha, SizeType m) {
  using dlaf::test::TypeUtilities;

  const bool op_a_lower = ((uplo == blas::Uplo::Lower && op == blas::Op::NoTrans) ||
                           (uplo == blas::Uplo::Upper && op != blas::Op::NoTrans));

  auto el_op_a = [op_a_lower, diag](const ElementIndex& index) {
    if ((op_a_lower && index.row() < index.col()) || (!op_a_lower && index.row() > index.col()) ||
        (diag == blas::Diag::Unit && index.row() == index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    double i = index.row();
    double k = index.col();

    return TypeUtilities<T>::polar((i + 1) / (k + .5), 2 * i - k);
  };

  auto el_b = [](const ElementIndex& index) {
    double k = index.row();
    double j = index.col();

    return TypeUtilities<T>::polar((k + .5) / (j + 2), k + j);
  };

  auto el_x = [m, alpha, diag, op_a_lower,
                                                el_b](const ElementIndex& index) {
    BaseType<T> kk = op_a_lower ? index.row() + 1 : m - index.row();

    double i = index.row();
    double j = index.col();
    T gamma = TypeUtilities<T>::polar((i + 1) / (j + 2), 2 * i + j);
    if (diag == blas::Diag::Unit)
      return ((kk - 1) * gamma + el_b(index)) * alpha;
    else
      return kk * gamma * alpha;
  };

  return std::make_tuple(el_op_a, el_b, el_x);
}

/// Returns a tuple of element generators of three matrices A(m x m), B (m x n), X (m x n), for which it
/// holds X = alpha B op(A) (n can be any value).
///
/// The elements of op(A) (@p el_op_a) are chosen such that:
///   op(A)_jk = (j+1) / (k+.5) * exp(I*(2*j-k)) for the referenced elements
///   op(A)_jk = -9.9 otherwise,
/// where I = 0 for real types or I is the complex unit for complex types.
///
/// The elements of B (@p el_B) are computed as
///   B_ik = (k+.5) / (i+2) * exp(I*(i+k)).
/// These data are typically used to check whether the result of the equation
/// performed with any algorithm is consistent with the computed values.
///
/// Finally, the elements of X (@p el_x) should be:
/// X_ij = (Sum_k B_ik * op(A)_kj) * alpha
///      = (B_ij * op(A)_jj + (kk-1) * gamma) * alpha,
/// where gamma = (j+1) / (i+2) * exp(I*(i+2*j)),
///       kk = j+1 if op(a) is an upper triangular matrix, or
///       kk = m-j if op(a) is an upper triangular matrix.
/// Therefore
/// X_ij = (B_ij + (kk-1) * gamma) * alpha, if diag == Unit
/// X_ij = kk * gamma * alpha, otherwise.
///
template <class ElementIndex, class T>
auto getRightTriangularMMSystem(blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha, SizeType n) {
  using dlaf::test::TypeUtilities;

  const bool op_a_lower = ((uplo == blas::Uplo::Lower && op == blas::Op::NoTrans) ||
                           (uplo == blas::Uplo::Upper && op != blas::Op::NoTrans));

  auto el_b = [](const ElementIndex& index) {
    double i = index.row();
    double k = index.col();

    return TypeUtilities<T>::polar((k + .5) / (i + 2), i + k);
  };

  auto el_op_a = [op_a_lower, diag](const ElementIndex& index) {
    if ((op_a_lower && index.row() < index.col()) || (!op_a_lower && index.row() > index.col()) ||
        (diag == blas::Diag::Unit && index.row() == index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    double k = index.row();
    double j = index.col();

    return TypeUtilities<T>::polar((j + 1) / (k + .5), 2 * j - k);
  };

  auto el_x = [n, alpha, diag, op_a_lower, el_b](const ElementIndex& index) {
    BaseType<T> kk = op_a_lower ? n - index.col() : index.col() + 1;

    double i = index.row();
    double j = index.col();
    T gamma = TypeUtilities<T>::polar((j + 1) / (i + 2), i + 2 * j);
    if (diag == blas::Diag::Unit)
      return ((kk - 1) * gamma + el_b(index)) * alpha;
    else
      return kk * gamma * alpha;
  };

  return std::make_tuple(el_op_a, el_b, el_x);
}
 
}
}
}
