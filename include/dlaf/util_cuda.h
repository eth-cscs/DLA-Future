//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#ifdef DLAF_WITH_CUDA

#include <cuComplex.h>
#include <complex>

namespace dlaf::util {
namespace internal {

template <typename T>
struct CppToCudaType {
  using type = T;
};

template <>
struct CppToCudaType<std::complex<float>> {
  using type = cuComplex;
};

template <>
struct CppToCudaType<std::complex<double>> {
  using type = cuDoubleComplex;
};

template <typename T>
struct CppToCudaType<const T> {
  using type = const typename CppToCudaType<T>::type;
};

template <typename T>
struct CppToCudaType<T*> {
  using type = typename CppToCudaType<T>::type*;
};

template <typename T>
using cppToCudaType_t = typename CppToCudaType<T>::type;
}

template <typename T>
constexpr typename internal::CppToCudaType<T*>::type cppToCudaCast(T* p) {
  return reinterpret_cast<typename internal::cppToCudaType_t<T*>>(p);
}

template <typename T>
constexpr typename internal::CppToCudaType<T>::type cppToCudaCast(T v) {
  return *cppToCudaCast(&v);
}

namespace cuda_operators {

// operators for Cuda types:
// The following operators are implemented:
// unary: -
// binary: +, -, *, /, * mixed real complex, / by real
// helper functions for real and complex types:
// conj, real, imag, fma
//
// Note: fma(a, b, c) is equivalent to a * b + c.
//       However for complex type it make a more efficient use of the fma instruction,
//       therefore it is preferable in performance critical kernels.

// Float
__host__ __device__ inline float conj(float a) {
  return a;
}

__host__ __device__ inline float real(float a) {
  return a;
}

__host__ __device__ inline float imag(float) {
  return 0.f;
}

__host__ __device__ inline float fma(float a, float b, float c) {
  return a * b + c;
}

// Complex
__host__ __device__ inline cuComplex operator-(cuComplex a) {
  return make_cuComplex(-a.x, -a.y);
}

__host__ __device__ inline cuComplex conj(cuComplex a) {
  return cuConjf(a);
}

__host__ __device__ inline float real(cuComplex a) {
  return a.x;
}

__host__ __device__ inline float imag(cuComplex a) {
  return a.y;
}

__host__ __device__ inline cuComplex operator+(cuComplex a, cuComplex b) {
  return cuCaddf(a, b);
}

__host__ __device__ inline cuComplex operator-(cuComplex a, cuComplex b) {
  return cuCsubf(a, b);
}

__host__ __device__ inline cuComplex operator*(cuComplex a, cuComplex b) {
  return cuCmulf(a, b);
}

__host__ __device__ inline cuComplex operator/(cuComplex a, cuComplex b) {
  return cuCdivf(a, b);
}

__host__ __device__ inline cuComplex fma(cuComplex a, cuComplex b, cuComplex c) {
  return cuCfmaf(a, b, c);
}

__host__ __device__ inline cuComplex operator*(float a, cuComplex b) {
  return make_cuComplex(a * b.x, a * b.y);
}

__host__ __device__ inline cuComplex operator*(cuComplex a, float b) {
  return operator*(b, a);
}

__host__ __device__ inline cuComplex operator/(cuComplex a, float b) {
  return make_cuComplex(a.x / b, a.y / b);
}

// Double
__host__ __device__ inline double conj(double a) {
  return a;
}

__host__ __device__ inline double real(double a) {
  return a;
}

__host__ __device__ inline double imag(double) {
  return 0.;
}

__host__ __device__ inline double fma(double a, double b, double c) {
  return a * b + c;
}

// Double complex
__host__ __device__ inline cuDoubleComplex operator-(cuDoubleComplex a) {
  return make_cuDoubleComplex(-a.x, -a.y);
}

__host__ __device__ inline cuDoubleComplex conj(cuDoubleComplex a) {
  return cuConj(a);
}

__host__ __device__ inline double real(cuDoubleComplex a) {
  return a.x;
}

__host__ __device__ inline double imag(cuDoubleComplex a) {
  return a.y;
}

__host__ __device__ inline cuDoubleComplex operator+(cuDoubleComplex a, cuDoubleComplex b) {
  return cuCadd(a, b);
}

__host__ __device__ inline cuDoubleComplex operator-(cuDoubleComplex a, cuDoubleComplex b) {
  return cuCsub(a, b);
}

__host__ __device__ inline cuDoubleComplex operator*(cuDoubleComplex a, cuDoubleComplex b) {
  return cuCmul(a, b);
}

__host__ __device__ inline cuDoubleComplex operator/(cuDoubleComplex a, cuDoubleComplex b) {
  return cuCdiv(a, b);
}

__host__ __device__ inline cuDoubleComplex fma(cuDoubleComplex a, cuDoubleComplex b, cuDoubleComplex c) {
  return cuCfma(a, b, c);
}

__host__ __device__ inline cuDoubleComplex operator*(double a, cuDoubleComplex b) {
  return make_cuDoubleComplex(a * b.x, a * b.y);
}

__host__ __device__ inline cuDoubleComplex operator*(cuDoubleComplex a, double b) {
  return operator*(b, a);
}

__host__ __device__ inline cuDoubleComplex operator/(cuDoubleComplex a, double b) {
  return make_cuDoubleComplex(a.x / b, a.y / b);
}

}
}

#endif
