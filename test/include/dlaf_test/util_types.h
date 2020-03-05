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
//using MatrixElementTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
 using MatrixElementTypes = ::testing::Types<float>;

using BufferTypes =
    ::testing::Types<int, const int, long long, const long long, float, const float, double,
                     const double, std::complex<float>, const std::complex<float>, std::complex<double>,
                     const std::complex<double>>;

template <class T>
struct TypeUtilities {
  /// @brief Returns r.
  static T element(double r, double i) {
    return static_cast<T>(r);
  }

  /// @brief Returns r.
  /// @pre r > 0
  static T polar(double r, double i) {
    return static_cast<T>(r);
  }

  /// @brief Returns val.
  static T conj(T val) {
    return val;
  }

  /// @brief Relative maximum error for a multiplication + addition.
  static constexpr T error = 2 * std::numeric_limits<T>::epsilon();
};

template <class T>
struct TypeUtilities<std::complex<T>> {
  /// @brief Returns r + I * i (I is the imaginary unit).
  static std::complex<T> element(double r, double i) {
    return std::complex<T>(static_cast<T>(r), static_cast<T>(i));
  }

  /// @brief Returns r * (cos(theta) + I * sin(theta)) (I is the imaginary unit).
  /// @pre r > 0
  static std::complex<T> polar(double r, double i) {
    return std::polar<T>(static_cast<T>(r), static_cast<T>(i));
  }

  /// @brief Returns std::conj(val).
  static std::complex<T> conj(std::complex<T> val) {
    return std::conj(val);
  }

  /// @brief Relative maximum error for a multiplication + addition.
  static constexpr T error = 8 * std::numeric_limits<T>::epsilon();
};

}
