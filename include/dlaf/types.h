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

template <class T>
struct TypeInfo;

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

}
