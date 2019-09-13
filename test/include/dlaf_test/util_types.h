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

#include <complex>
#include "gtest/gtest.h"

namespace dlaf_test {

using ElementTypes =
    ::testing::Types<int, long long, float, double, std::complex<float>, std::complex<double>>;
using MatrixElementTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;

template <class T>
struct TypeUtilities {
  /// Returns r.
  static T element(T r, T i) {
    return r;
  }
};

template <class T>
struct TypeUtilities<std::complex<T>> {
  /// Returns r + I * i (I is the imaginary unit).
  static std::complex<T> element(T r, T i) {
    return std::complex<T>(r, i);
  }
};

}
