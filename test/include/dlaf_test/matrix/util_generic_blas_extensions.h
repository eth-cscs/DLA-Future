//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <functional>
#include <tuple>

#include <blas.hh>

#include <dlaf/common/assert.h>
#include <dlaf/matrix/index.h>
#include <dlaf/types.h>

#include <dlaf_test/matrix/util_generic_blas.h>
#include <dlaf_test/util_types.h>

/// @file

// https://github.com/eth-cscs/DLA-Future/blob/master/include/dlaf/blas/tile_extensions.h#L56-L71
// https://github.com/eth-cscs/DLA-Future/blob/master/include/dlaf/blas/tile_extensions.h#L100

namespace dlaf::matrix::test {

template <class ElementIndex, class T>
auto getMatrixScal(const T beta) {
  using dlaf::test::TypeUtilities;

  auto el_a = [](const ElementIndex& index) {
    const double i = index.row();
    const double j = index.col();
    return TypeUtilities<T>::polar(.9 * (i + 1) / (j + .5), 2 * i - j);
  };

  auto res_a = [beta, el_a](const ElementIndex& index) { return beta * el_a(index); };

  return std::make_tuple(el_a, res_a);
}

template <class ElementIndex, class T>
auto getMatrixAdd(const T alpha) {
  using dlaf::test::TypeUtilities;

  auto el_a = [](const ElementIndex& index) {
    const double i = index.row();
    const double j = index.col();
    return TypeUtilities<T>::polar(.9 * (i + 1) / (j + .5), 2 * i - j);
  };

  auto el_b = [](const ElementIndex& index) {
    const double i = index.row();
    const double j = index.col();
    return TypeUtilities<T>::polar(1.2 * i / (j + 1), -i + j);
  };

  auto res_a = [alpha, el_a, el_b](const ElementIndex& index) {
    return el_a(index) + alpha * el_b(index);
  };

  return std::make_tuple(el_a, el_b, res_a);
}
}
