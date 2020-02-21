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

namespace dlaf {

using SizeType = int;

enum class Device { CPU, GPU };

enum class Execution { Default, MC, GPU };

template <class T>
struct TypeInfo;

template <class T>
struct TypeInfo<const T> : public TypeInfo<T> {};

template <>
struct TypeInfo<float> {
  using BaseType = float;
  using Type = float;
  using ComplexType = std::complex<float>;
};

template <>
struct TypeInfo<double> {
  using BaseType = double;
  using Type = double;
  using ComplexType = std::complex<double>;
};

template <>
struct TypeInfo<std::complex<float>> {
  using BaseType = float;
  using Type = std::complex<float>;
  using ComplexType = std::complex<float>;
};

template <>
struct TypeInfo<std::complex<double>> {
  using BaseType = double;
  using Type = std::complex<double>;
  using ComplexType = std::complex<double>;
};

template <class T>
using BaseType = typename TypeInfo<T>::BaseType;

template <class T>
using ComplexType = typename TypeInfo<T>::ComplexType;

/// Return complex conjugate of a complex number
template <class T>
std::complex<T> conj(const std::complex<T> number) {
  return std::conj(number);
}

/// Return complex conjugate of a real number as a real number
///
/// It differs from std::conj just in the return type. In fact,
/// std::conj always returns a std::complex
template <class T>
T conj(const T number) {
  return number;
}
}
