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

/// @file

#include "blas.hh"
#include "dlaf/matrix.h"
#include "dlaf_test/matrix/util_generic_blas.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/util_types.h"

namespace dlaf {
namespace matrix {
namespace test {
using namespace dlaf_test;

/// Sets the elements of the matrix.
///
/// The (i, j)-element of the matrix is set to el({i, j}) if op == NoTrans,
///                                          el({j, i}) if op == Trans,
///                                          conj(el({j, i})) if op == ConjTrans.
/// @pre el is a callable with an argument of type const GlobalElementIndex& or GlobalElementIndex and
/// return type T.
template <template <class, Device> class MatrixType, class T, class ElementGetter>
void set(MatrixType<T, Device::CPU>& mat, ElementGetter el, blas::Op op) {
  switch (op) {
    case blas::Op::NoTrans:
      set(mat, el);
      break;

    case blas::Op::Trans: {
      auto op_el = [&el](GlobalElementIndex i) {
        i.transpose();
        return el(i);
      };
      set(mat, op_el);
      break;
    }

    case blas::Op::ConjTrans: {
      auto op_el = [&el](GlobalElementIndex i) {
        i.transpose();
        return TypeUtilities<T>::conj(el(i));
      };
      set(mat, op_el);
      break;
    }
  }
}
}
}
}
