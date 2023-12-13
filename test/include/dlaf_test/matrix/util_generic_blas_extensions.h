//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
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

#include <dlaf_test/util_types.h>

/// @file

namespace dlaf::matrix::test {

namespace internal {
template <class ElementGetter>
auto opValFunc(ElementGetter&& val, const blas::Op op) {
  std::function op_val = val;
  switch (op) {
    case blas::Op::NoTrans:
      break;
    case blas::Op::Trans: {
      op_val = [&val](auto i) {
        i.transpose();
        return val(i);
      };
      break;
    }
    case blas::Op::ConjTrans: {
      op_val = [&val](auto i) {
        i.transpose();
        return dlaf::conj(val(i));
      };
      break;
    }
  }
  return op_val;
}
}

template <class ElementIndex, class T>
auto getMatrixScal(const T beta) {
  using dlaf::test::TypeUtilities;

  auto el_a = [](const ElementIndex& index) {
    const double i = index.row();
    const double k = index.col();
    return TypeUtilities<T>::polar(.9 * (i + 1) / (k + .5), 2 * i - k);
  };

  auto res_a = [beta, el_a](const ElementIndex& index) {
    const double i = index.row();
    const double j = index.col();
    return beta * el_op_a(index);
  };

  using internal::opValFunc;
  return std::make_tuple(el_a, res_a);
}
}
