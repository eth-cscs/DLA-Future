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

#include <complex>
#include "gtest/gtest.h"

namespace dlaf {
namespace test {

using ElementTypes =
    ::testing::Types<int, long long, float, double, std::complex<float>, std::complex<double>>;
//using MatrixElementTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
 using MatrixElementTypes = ::testing::Types<float, double>;
 
template <class T>
struct TypeUtilities {
  /// Returns r.
  static constexpr T element(double r, double /* i */) {
    return static_cast<T>(r);
  }

  /// Returns r.
  ///
  /// @pre r > 0.
  static constexpr T polar(double r, double /* theta */) {
    return static_cast<T>(r);
  }

  /// Returns val.
  static constexpr T conj(T val) {
    return val;
  }

  /// Relative maximum error for a multiplication + addition.
  static constexpr T error = 2 * std::numeric_limits<T>::epsilon();
};

template <class T>
constexpr T TypeUtilities<T>::error;

template <class T>
struct TypeUtilities<std::complex<T>> {
  /// Returns r + I * i (I is the imaginary unit).
  static constexpr std::complex<T> element(double r, double i) {
    return std::complex<T>(static_cast<T>(r), static_cast<T>(i));
  }

  /// Returns r * (cos(theta) + I * sin(theta)) (I is the imaginary unit).
  ///
  /// @pre r > 0.
  static constexpr std::complex<T> polar(double r, double theta) {
    return std::polar<T>(static_cast<T>(r), static_cast<T>(theta));
  }

  /// Returns std::conj(val).
  static constexpr std::complex<T> conj(std::complex<T> val) {
    return std::conj(val);
  }

  /// Relative maximum error for a multiplication + addition.
  static constexpr T error = 8 * std::numeric_limits<T>::epsilon();
};

template <class T>
constexpr T TypeUtilities<std::complex<T>>::error;
}
}
