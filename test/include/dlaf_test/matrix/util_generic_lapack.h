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

/// Returns a tuple of element generators of three matrices T (m x m), A(m x m) and B (m x m).
///
/// The elements of the Hermitian matrix A (@p el_a) are chosen such that for @p itype = 1:
/// A_ij = (i+1)(j+1) (beta*beta*gamma)/(2^(i+j))*exp(I*alpha*(i-j))
/// or for @p itype = 2 and @p itype = 3:
/// A_ij = gamma/(2^(i+j))*exp(I*alpha*(i-j))
/// where alpha = 0 and I = 0 for real types or I is the complex unit for complex types.
///
/// The elements of T (@p el_T), a triangular matrix, are computed, for @p itype = 1, as
/// T_ij = beta/(2^(i-j)) * exp(I*alpha*(i-j))
/// and, for @p itype = 2 or @p itype = 3, as
/// T_ij = beta/(2^(-i-j)) * exp(I*alpha*(i-j))
/// When the matrix is @p Upper row and column indices are swapped.
///
/// Finally, the elements of B (@p el_b) should be, for @p itype = 1:
/// B_ij = (gamma)/(2^(i+j))*exp(I*alpha*(i-j))
/// and for @p itype = 2 or @p itype = 3:
/// B_ij = ((n-i)*(n-j)*(beta*beta*gamma)/(2^(-i-j))*exp(I*alpha*(i-j))
/// where @p n is the size of the matrix.
///
template <class ElementIndex, class T>
auto getGeneralizedEigenvalueSystem(SizeType n, int itype, blas::Uplo uplo, T alpha, T beta, T gamma) {
  std::function<T(const ElementIndex&)> el_t = [itype, uplo, alpha, beta](const ElementIndex& index) {
    if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
        (uplo == blas::Uplo::Upper && index.row() > index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    double i, j;
    if (uplo == blas::Uplo::Lower) {
      i = index.row();
      j = index.col();
    }
    else if (uplo == blas::Uplo::Upper) {
      j = index.row();
      i = index.col();
    }
    else if (uplo == blas::Uplo::General) {
      std::cout << "uplo == General not allowed" << std::endl;
      std::abort;
    }

    if (itype == 1) {
      return TypeUtilities<T>::polar(beta / std::exp2(i - j), alpha * (i - j));
    }
    else if (itype == 2 || itype == 3) {
      return TypeUtilities<T>::polar(beta / std::exp2(-i - j), alpha * (i - j));
    }
    else {
      std::cout << "HEGST: itype > 3 now allowed" << std::endl;
      std::abort;
    }
  };

  std::function<T(const ElementIndex&)> el_a = [itype, uplo, alpha, beta,
                                                gamma](const ElementIndex& index) {
    if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
        (uplo == blas::Uplo::Upper && index.row() > index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    double i = index.row();
    double j = index.col();
    if (itype == 1) {
      return TypeUtilities<T>::polar((i + 1) * (j + 1) * (beta * beta * gamma) / std::exp2(i + j),
                                     alpha * (i - j));
    }
    else if (itype == 2 || itype == 3) {
      return TypeUtilities<T>::polar(gamma / std::exp2(i + j), alpha * (i - j));
    }
    else {
      std::cout << "HEGST: itype > 3 now allowed" << std::endl;
      std::abort;
    }
  };

  std::function<T(const ElementIndex&)> el_b = [n, itype, uplo, alpha, beta,
                                                gamma](const ElementIndex& index) {
    if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
        (uplo == blas::Uplo::Upper && index.row() > index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    double i = index.row();
    double j = index.col();

    if (itype == 1) {
      return TypeUtilities<T>::polar(gamma / std::exp2(i + j), alpha * (i - j));
    }
    else if (itype == 2 || itype == 3) {
      return TypeUtilities<T>::polar((n - i) * (n - j) * (beta * beta * gamma) / std::exp2(-i - j),
                                     alpha * (i - j));
    }
    else {
      std::cout << "HEGST: itype > 3 now allowed" << std::endl;
      std::abort;
    }
  };

  return std::make_tuple(el_t, el_a, el_b);
}

}
}
}
