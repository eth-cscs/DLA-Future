//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
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

using namespace dlaf_test;

/// Returns elements of the three matrices, solving the triangular matrix equation
/// op(A) B = X (with A on side == Left).
///
/// The tile elements of A matrix (el_op_a) are chosen such that:
///   op(A)_ik = (i+1) / (k+.5) * exp(I*(2*i-k)) for the referenced elements
///   op(A)_ik = -9.9 otherwise.
///
/// The elements of B (el_b) should be:
/// B_ij = (Sum_k op(A)_ik * res_Xkj) / alpha
///      = (op(A)_ii * res_Xij + (kk-1) * gamma) / alpha,
/// where gamma = (i+1) / (j+2) * exp(I*(2*i+j)),
///       kk = i+1 if op(a) is an lower triangular matrix, or
///       kk = m-i if op(a) is an lower triangular matrix.
/// Therefore
/// B_ij = (res_Xij + (kk-1) * gamma) / alpha, if diag == Unit
/// B_ij = kk * gamma / alpha, otherwise.
///
/// Finally, the X matrix elements (res_b) are computed as
///   res_Xkj = (k+.5) / (j+2) * exp(I*(k+j)),
/// where I = 0 for real types or I is the complex unit for complex types.
/// These data are typically used to check whether the result of the equation
/// performed with any algorithm is consistent with the computed values.
///
template <class ElementIndex, class T>
auto getLeftTriangularSystem(blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha, SizeType m) {
  bool op_a_lower = false;
  if ((uplo == blas::Uplo::Lower && op == blas::Op::NoTrans) ||
      (uplo == blas::Uplo::Upper && op != blas::Op::NoTrans))
    op_a_lower = true;

  std::function<T(const ElementIndex&)> el_op_a = [op_a_lower, diag](const ElementIndex& index) {
    if ((op_a_lower && index.row() < index.col()) || (!op_a_lower && index.row() > index.col()) ||
        (diag == blas::Diag::Unit && index.row() == index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    double i = index.row();
    double k = index.col();

    return TypeUtilities<T>::polar((i + 1) / (k + .5), 2 * i - k);
  };

  std::function<T(const ElementIndex&)> res_b = [](const ElementIndex& index) {
    double k = index.row();
    double j = index.col();

    return TypeUtilities<T>::polar((k + .5) / (j + 2), k + j);
  };

  std::function<T(const ElementIndex&)> el_b = [m, alpha, diag, op_a_lower,
                                                res_b](const ElementIndex& index) {
    BaseType<T> kk = op_a_lower ? index.row() + 1 : m - index.row();

    double i = index.row();
    double j = index.col();
    T gamma = TypeUtilities<T>::polar((i + 1) / (j + 2), 2 * i + j);
    if (diag == blas::Diag::Unit)
      return ((kk - 1) * gamma + res_b(index)) / alpha;
    else
      return kk * gamma / alpha;
  };

  return std::make_tuple(el_op_a, el_b, res_b);
}

/// Returns elements of the three matrices, solving the triangular matrix equation
/// B op(A) = X (with A on side == right)
///
/// The tile elements of A matrix (el_op_a) are chosen such that:
///   op(A)_kj = (j+1) / (k+.5) * exp(I*(2*j-k)) for the referenced elements
///   op(A)_kj = -9.9 otherwise.
///
/// The elements of B (el_b) should be:
/// B_ij = (Sum_k res_Xik * op(A)_kj) / alpha
///      = (res_Xij * op(A)_jj + (kk-1) * gamma) / alpha,
/// where gamma = (j+1) / (i+2) * exp(I*(i+2*j)),
///       kk = j+1 if op(a) is an upper triangular matrix, or
///       kk = m-j if op(a) is an upper triangular matrix.
/// Therefore
/// B_ij = (res_Xij + (kk-1) * gamma) / alpha, if diag == Unit
/// B_ij = kk * gamma / alpha, otherwise.
///
/// Finally, the X matrix elements (res_b) are computed as
///   res_Xik = (k+.5) / (i+2) * exp(I*(i+k)),
/// where I = 0 for real types or I is the complex unit for complex types.
/// These data are typically used to check whether the result of the equation
/// performed with any algorithm is consistent with the computed values.
///
template <class ElementIndex, class T>
auto getRightTriangularSystem(blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha, SizeType n) {
  bool op_a_lower = false;
  if ((uplo == blas::Uplo::Lower && op == blas::Op::NoTrans) ||
      (uplo == blas::Uplo::Upper && op != blas::Op::NoTrans))
    op_a_lower = true;

  auto res_b = [](const ElementIndex& index) {
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

  auto el_b = [n, alpha, diag, op_a_lower, res_b](const ElementIndex& index) {
    BaseType<T> kk = op_a_lower ? n - index.col() : index.col() + 1;

    double i = index.row();
    double j = index.col();
    T gamma = TypeUtilities<T>::polar((j + 1) / (i + 2), i + 2 * j);
    if (diag == blas::Diag::Unit)
      return ((kk - 1) * gamma + res_b(index)) / alpha;
    else
      return kk * gamma / alpha;
  };

  return std::make_tuple(el_op_a, el_b, res_b);
}

}
}
}
