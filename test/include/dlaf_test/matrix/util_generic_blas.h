//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <functional>
#include <tuple>

#include <blas.hh>

#include "dlaf/common/assert.h"
#include "dlaf/matrix/index.h"
#include "dlaf/types.h"
#include "dlaf_test/util_types.h"

/// @file

namespace dlaf::matrix::test {

template <class T>
auto getSubMatrixMatrixMultiplication(const SizeType a, const SizeType b, const SizeType m,
                                      const SizeType n, const SizeType k, const T alpha, const T beta,
                                      const blas::Op opA, const blas::Op opB) {
  using dlaf::test::TypeUtilities;

  if (opA != blas::Op::NoTrans)
    DLAF_UNIMPLEMENTED(opA);
  if (opB != blas::Op::NoTrans)
    DLAF_UNIMPLEMENTED(opB);

  DLAF_ASSERT(a >= 0 and a < m and a < n, a, m, n);
  DLAF_ASSERT(b >= 0 and b < m and b < n, b, m, n);
  DLAF_ASSERT(a <= b, a, b);

  auto elA = [a, b](const GlobalElementIndex ik) {
    if (ik.row() < a || ik.row() > b || ik.col() < a || ik.col() > b)
      return TypeUtilities<T>::polar(13, -26);

    const double i = ik.row();
    const double k = ik.col();

    return TypeUtilities<T>::polar((i + 1) / (k + .5), 2 * i - k);
  };

  auto elB = [a, b](const GlobalElementIndex kj) {
    if (kj.row() < a || kj.row() > b || kj.col() < a || kj.col() > b)
      return TypeUtilities<T>::polar(13, -26);

    const double k = kj.row();
    const double j = kj.col();

    return TypeUtilities<T>::polar((k + .5) / (j + 2), k + j);
  };

  auto elC = [a, b](const GlobalElementIndex) { return TypeUtilities<T>::polar(0, 0); };

  auto elR = [a, b, k, alpha, beta](const GlobalElementIndex ij) {
    if (ij.row() < a || ij.row() > b || ij.col() < a || ij.col() > b)
      return TypeUtilities<T>::polar(13, -26);

    const double i = ij.row();
    const double j = ij.col();

    return alpha * TypeUtilities<T>::polar((i + 1) / (j + 2) * (b - a + 1), (2 * i) + j);
  };

  return std::make_tuple<>(elA, elB, elC, elR);
}

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
///       kk = m-i if op(a) is an upper triangular matrix.
/// Therefore
/// B_ij = (X_ij + (kk-1) * gamma) / alpha, if diag == Unit,
/// B_ij = kk * gamma / alpha, otherwise.
///
template <class ElementIndex, class T>
auto getLeftTriangularSystem(blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha, SizeType m) {
  using dlaf::test::TypeUtilities;
  using FuncType = std::function<T(const ElementIndex&)>;

  const bool op_a_lower = ((uplo == blas::Uplo::Lower && op == blas::Op::NoTrans) ||
                           (uplo == blas::Uplo::Upper && op != blas::Op::NoTrans));

  FuncType el_op_a = [op_a_lower, diag](const ElementIndex& index) {
    if ((op_a_lower && index.row() < index.col()) || (!op_a_lower && index.row() > index.col()) ||
        (diag == blas::Diag::Unit && index.row() == index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    const double i = index.row();
    const double k = index.col();

    return TypeUtilities<T>::polar((i + 1) / (k + .5), 2 * i - k);
  };

  FuncType el_x = [](const ElementIndex& index) {
    const double k = index.row();
    const double j = index.col();

    return TypeUtilities<T>::polar((k + .5) / (j + 2), k + j);
  };

  FuncType el_b = [m, alpha, diag, op_a_lower, el_x](const ElementIndex& index) {
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

/// Returns a tuple of element generators of three matrices A(n x n), B (m x n), X (m x n), for which it
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
///       kk = n-j if op(a) is an lower triangular matrix.
/// Therefore
/// B_ij = (X_ij + (kk-1) * gamma) / alpha, if diag == Unit,
/// B_ij = kk * gamma / alpha, otherwise.
///
template <class ElementIndex, class T>
auto getRightTriangularSystem(blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha, SizeType n) {
  using dlaf::test::TypeUtilities;
  using FuncType = std::function<T(const ElementIndex&)>;

  const bool op_a_lower = ((uplo == blas::Uplo::Lower && op == blas::Op::NoTrans) ||
                           (uplo == blas::Uplo::Upper && op != blas::Op::NoTrans));

  FuncType el_x = [](const ElementIndex& index) {
    const double i = index.row();
    const double k = index.col();

    return TypeUtilities<T>::polar((k + .5) / (i + 2), i + k);
  };

  FuncType el_op_a = [op_a_lower, diag](const ElementIndex& index) {
    if ((op_a_lower && index.row() < index.col()) || (!op_a_lower && index.row() > index.col()) ||
        (diag == blas::Diag::Unit && index.row() == index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    const double k = index.row();
    const double j = index.col();

    return TypeUtilities<T>::polar((j + 1) / (k + .5), 2 * j - k);
  };

  FuncType el_b = [n, alpha, diag, op_a_lower, el_x](const ElementIndex& index) {
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

/// Dispatcher function to choose between left and right implementation of triangular system generator function
template <class ElementIndex, class T>
auto getTriangularSystem(blas::Side side, blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha,
                         SizeType m, SizeType n) {
  if (side == blas::Side::Left)
    return dlaf::matrix::test::getLeftTriangularSystem<ElementIndex, T>(uplo, op, diag, alpha, m);
  else
    return dlaf::matrix::test::getRightTriangularSystem<ElementIndex, T>(uplo, op, diag, alpha, n);
}

}
