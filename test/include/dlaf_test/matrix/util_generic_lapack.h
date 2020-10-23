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
auto getGenToStdElementSetters(SizeType n, int itype, blas::Uplo uplo, T alpha, T beta, T gamma) {
  DLAF_ASSERT(uplo == blas::Uplo::Lower || uplo == blas::Uplo::Upper, "Only Upper and Lower supported",
              uplo);
  DLAF_ASSERT(itype > 0 && itype < 4, "Only itype = 1, 2, 3 allowed", itype);

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

}
}
}
