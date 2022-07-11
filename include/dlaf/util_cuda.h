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

#ifdef DLAF_WITH_GPU

#if defined DLAF_WITH_HIP

// Float
#define make_cuComplex make_hipComplex
#define cuCaddf        hipCaddf
#define cuCsubf        hipCsubf
#define cuCmulf        hipCmulf
#define cuCdivf        hipCdivf
#define cuCfmaf        hipCfmaf

// Double
#define make_cuDoubleComplex make_hipDoubleComplex
#define cuCadd               hipCadd
#define cuCsub               hipCsub
#define cuCmul               hipCmul
#define cuCdiv               hipCdiv
#define cuCfma               hipCfma

#endif

#include <complex>
#include "dlaf/gpu/blas/api.h"

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
//       However for complex type it makes a more efficient use of the fma instruction,
//       therefore it is preferable in performance critical kernels.

__host__ __device__ inline constexpr unsigned ceilDiv(unsigned i, unsigned j) noexcept {
  return (i + j - 1) / j;
}

// Float
__host__ __device__ inline float conj(const float& a) noexcept {
  return a;
}

__host__ __device__ inline float real(const float& a) noexcept {
  return a;
}

__host__ __device__ inline float imag(const float&) noexcept {
  return 0.f;
}

__host__ __device__ inline float fma(const float& a, const float& b, const float& c) noexcept {
  return a * b + c;
}

// Complex
__host__ __device__ inline cuComplex operator-(const cuComplex& a) noexcept {
  return make_cuComplex(-a.x, -a.y);
}

__host__ __device__ inline cuComplex conj(const cuComplex& a) noexcept {
#ifdef DLAF_WITH_CUDA
  return cuConjf(a);
#elif defined DLAF_WITH_HIP
  return make_cuComplex(a.x, -a.y);
#endif
}

__host__ __device__ inline float real(const cuComplex& a) noexcept {
  return a.x;
}

__host__ __device__ inline float imag(const cuComplex& a) noexcept {
  return a.y;
}

#if defined DLAF_WITH_CUDA
__host__ __device__ inline bool operator==(const cuComplex& a, const cuComplex& b) noexcept {
  return a.x == b.x && a.y == b.y;
}

__host__ __device__ inline bool operator!=(const cuComplex& a, const cuComplex& b) noexcept {
  return !operator==(a, b);
}
#endif

__host__ __device__ inline cuComplex operator+(const cuComplex& a, const cuComplex& b) noexcept {
  return cuCaddf(a, b);
}

__host__ __device__ inline cuComplex operator-(const cuComplex& a, const cuComplex& b) noexcept {
  return cuCsubf(a, b);
}

__host__ __device__ inline cuComplex operator*(const cuComplex& a, const cuComplex& b) noexcept {
  return cuCmulf(a, b);
}

__host__ __device__ inline cuComplex operator/(const cuComplex& a, const cuComplex& b) noexcept {
  return cuCdivf(a, b);
}

__host__ __device__ inline cuComplex fma(const cuComplex& a, const cuComplex& b,
                                         const cuComplex& c) noexcept {
  return cuCfmaf(a, b, c);
}

__host__ __device__ inline cuComplex operator*(const float& a, const cuComplex& b) noexcept {
  return make_cuComplex(a * b.x, a * b.y);
}

__host__ __device__ inline cuComplex operator*(const cuComplex& a, const float& b) noexcept {
  return operator*(b, a);
}

__host__ __device__ inline cuComplex operator/(const cuComplex& a, const float& b) noexcept {
  return make_cuComplex(a.x / b, a.y / b);
}

// Double
__host__ __device__ inline double conj(const double& a) noexcept {
  return a;
}

__host__ __device__ inline double real(const double& a) noexcept {
  return a;
}

__host__ __device__ inline double imag(const double&) noexcept {
  return 0.;
}

__host__ __device__ inline double fma(const double& a, const double& b, const double& c) noexcept {
  return a * b + c;
}

// Double complex
__host__ __device__ inline cuDoubleComplex operator-(const cuDoubleComplex& a) noexcept {
  return make_cuDoubleComplex(-a.x, -a.y);
}

__host__ __device__ inline cuDoubleComplex conj(const cuDoubleComplex& a) noexcept {
#if defined DLAF_WITH_CUDA
  return cuConj(a);
#elif defined DLAF_WITH_HIP
  return make_cuDoubleComplex(a.x, -a.y);
#endif
}

__host__ __device__ inline double real(const cuDoubleComplex& a) noexcept {
  return a.x;
}

__host__ __device__ inline double imag(const cuDoubleComplex& a) noexcept {
  return a.y;
}

#if defined DLAF_WITH_CUDA
__host__ __device__ inline bool operator==(const cuDoubleComplex& a, const cuDoubleComplex& b) noexcept {
  return a.x == b.x && a.y == b.y;
}

__host__ __device__ inline bool operator!=(const cuDoubleComplex& a, const cuDoubleComplex& b) noexcept {
  return !operator==(a, b);
}
#endif

__host__ __device__ inline cuDoubleComplex operator+(const cuDoubleComplex& a,
                                                     const cuDoubleComplex& b) noexcept {
  return cuCadd(a, b);
}

__host__ __device__ inline cuDoubleComplex operator-(const cuDoubleComplex& a,
                                                     const cuDoubleComplex& b) noexcept {
  return cuCsub(a, b);
}

__host__ __device__ inline cuDoubleComplex operator*(const cuDoubleComplex& a,
                                                     const cuDoubleComplex& b) noexcept {
  return cuCmul(a, b);
}

__host__ __device__ inline cuDoubleComplex operator/(const cuDoubleComplex& a,
                                                     const cuDoubleComplex& b) noexcept {
  return cuCdiv(a, b);
}

__host__ __device__ inline cuDoubleComplex fma(const cuDoubleComplex& a, const cuDoubleComplex& b,
                                               const cuDoubleComplex& c) noexcept {
  return cuCfma(a, b, c);
}

__host__ __device__ inline cuDoubleComplex operator*(const double& a,
                                                     const cuDoubleComplex& b) noexcept {
  return make_cuDoubleComplex(a * b.x, a * b.y);
}

__host__ __device__ inline cuDoubleComplex operator*(const cuDoubleComplex& a,
                                                     const double& b) noexcept {
  return operator*(b, a);
}

__host__ __device__ inline cuDoubleComplex operator/(const cuDoubleComplex& a,
                                                     const double& b) noexcept {
  return make_cuDoubleComplex(a.x / b, a.y / b);
}

}
}

#endif
