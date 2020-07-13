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

/// Returns a tuple of element generators of three matrices A(m x m), L (m x m), L**H (m x m), B (m x m)
/// for which it holds inv(L) * A * inv(L**H) = B
///
/// The elements of the Hermitian matrix A (@p el_a) are chosen such that:
/// A_ij = (i+1)(j+1) (beta*beta*gamma)/(2^(i+j))*exp(I*alpha*(i-j))
/// where alpha = 0 and I = 0 for real types or I is the complex unit for complex types.
///
/// The elements of L (@p el_l) are computed as
/// L_ij = beta/(2^(i-j)) * exp(I*alpha*(i-j))
/// where i > j and I = 0 for real types or I is the complex unit for complex types.
///
/// Finally, the elements of B (@p el_b) should be:
/// B_ij = (gamma)/(2^(i+j))*exp(I*alpha*(i-j))
///
template <class ElementIndex, class T>
auto getLowerHermitianSystem(T alpha, T beta, T gamma) {
  std::function<T(const ElementIndex&)> el_l = [alpha, beta](const ElementIndex& index) {
    if (index.row() < index.col())
      return TypeUtilities<T>::element(-9.9, 0);

    double i = index.row();
    double j = index.col();

    return TypeUtilities<T>::polar(beta / std::exp2(i - j), alpha * (i - j));
  };

  std::function<T(const ElementIndex&)> el_a = [alpha, beta, gamma](const ElementIndex& index) {
    if (index.row() < index.col())
      return TypeUtilities<T>::element(-9.9, 0);

    double i = index.row();
    double j = index.col();

    return TypeUtilities<T>::polar((i + 1) * (j + 1) * (beta * beta * gamma) / std::exp2(i + j),
                                   alpha * (i - j));
  };

  std::function<T(const ElementIndex&)> res_b = [alpha, gamma](const ElementIndex& index) {
    if (index.row() < index.col())
      return TypeUtilities<T>::element(-9.9, 0);

    double i = index.row();
    double j = index.col();

    return TypeUtilities<T>::polar(gamma / std::exp2(i + j), alpha * (i - j));
  };

  return std::make_tuple(el_l, el_a, res_b);
}

}
}
}
