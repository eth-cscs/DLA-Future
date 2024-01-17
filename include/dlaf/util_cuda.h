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

#ifdef DLAF_WITH_GPU

#ifdef DLAF_WITH_HIP

// Float
#define make_cuComplex make_hipComplex
#define cuConjf        hipConjf
#define cuCaddf        hipCaddf
#define cuCsubf        hipCsubf
#define cuCmulf        hipCmulf
#define cuCdivf        hipCdivf
#define cuCfmaf        hipCfmaf

// Double
#define make_cuDoubleComplex make_hipDoubleComplex
#define cuConj               hipConj
#define cuCadd               hipCadd
#define cuCsub               hipCsub
#define cuCmul               hipCmul
#define cuCdiv               hipCdiv
#define cuCfma               hipCfma

#endif

#include <complex>

#include <dlaf/gpu/blas/api.h>

namespace dlaf::util {

#if defined(DLAF_WITH_HIP)
struct hipComplexWrapper {
    hipComplex v{};
    __host__ __device__ hipComplexWrapper(const hipComplex& v) : v(v) {}
};

struct hipDoubleComplexWrapper {
    hipDoubleComplex v{};
    __host__ __device__ hipDoubleComplexWrapper(const hipDoubleComplex& v) : v(v) {}
};
#endif

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

#if defined(DLAF_WITH_CUDA)
template <typename T>
struct CppToCudaWrapperType {
  using type = T;
};

template <>
struct CppToCudaWrapperType<std::complex<float>> {
  using type = cuComplex;
};

template <>
struct CppToCudaWrapperType<std::complex<double>> {
  using type = cuDoubleComplex;
};

template <typename T>
struct CppToCudaWrapperType<const T> {
  using type = const typename CppToCudaWrapperType<T>::type;
};

template <typename T>
struct CppToCudaWrapperType<T*> {
  using type = typename CppToCudaWrapperType<T>::type*;
};

template <typename T>
using cppToCudaWrapperType_t = typename CppToCudaWrapperType<T>::type;
#elif defined(DLAF_WITH_HIP)
template <typename T>
struct CppToCudaWrapperType {
  using type = T;
};

template <>
struct CppToCudaWrapperType<std::complex<float>> {
  using type = hipComplexWrapper;
};

template <>
struct CppToCudaWrapperType<std::complex<double>> {
  using type = hipDoubleComplexWrapper;
};

template <typename T>
struct CppToCudaWrapperType<const T> {
  using type = const typename CppToCudaWrapperType<T>::type;
};

template <typename T>
struct CppToCudaWrapperType<T*> {
  using type = typename CppToCudaWrapperType<T>::type*;
};

template <typename T>
using cppToCudaWrapperType_t = typename CppToCudaWrapperType<T>::type;
#endif
}

template <typename T>
constexpr typename internal::CppToCudaType<T*>::type cppToCudaCast(T* p) {
  return reinterpret_cast<typename internal::cppToCudaType_t<T*>>(p);
}

template <typename T>
constexpr typename internal::CppToCudaType<T>::type cppToCudaCast(T v) {
  return *cppToCudaCast(&v);
}

template <typename T>
constexpr typename internal::CppToCudaWrapperType<T*>::type cppToCudaWrapperCast(T* p) {
  return reinterpret_cast<typename internal::cppToCudaWrapperType_t<T*>>(p);
}

template <typename T>
constexpr typename internal::CppToCudaWrapperType<T>::type cppToCudaWrapperCast(T v) {
  return *cppToCudaWrapperCast(&v);
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
#ifdef DLAF_WITH_CUDA
__host__ __device__ inline cuComplex conj(const cuComplex& a) noexcept {
  return cuConjf(a);
}

__host__ __device__ inline float real(const cuComplex& a) noexcept {
  return a.x;
}

__host__ __device__ inline float imag(const cuComplex& a) noexcept {
  return a.y;
}

__host__ __device__ inline cuComplex fma(const cuComplex& a, const cuComplex& b,
                                         const cuComplex& c) noexcept {
  return cuCfmaf(a, b, c);
}

__host__ __device__ inline cuComplex operator-(const cuComplex& a) noexcept {
  return make_cuComplex(-a.x, -a.y);
}

__host__ __device__ inline bool operator==(const cuComplex& a, const cuComplex& b) noexcept {
  return a.x == b.x && a.y == b.y;
}

__host__ __device__ inline bool operator!=(const cuComplex& a, const cuComplex& b) noexcept {
  return !(operator==)(a, b);
}

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

__host__ __device__ inline cuComplex operator*(const float& a, const cuComplex& b) noexcept {
  return make_cuComplex(a * b.x, a * b.y);
}

__host__ __device__ inline cuComplex operator*(const cuComplex& a, const float& b) noexcept {
  return (operator*)(b, a);
}

__host__ __device__ inline cuComplex operator/(const cuComplex& a, const float& b) noexcept {
  return make_cuComplex(a.x / b, a.y / b);
}
#elif defined(DLAF_WITH_HIP)
__host__ __device__ inline hipComplexWrapper conj(const hipComplexWrapper& a) noexcept {
  return cuConjf(a.v);
}

__host__ __device__ inline float real(const hipComplexWrapper& a) noexcept {
  return a.v.x;
}

__host__ __device__ inline float imag(const hipComplexWrapper& a) noexcept {
  return a.v.y;
}

__host__ __device__ inline hipComplexWrapper fma(const hipComplexWrapper& a, const hipComplexWrapper& b,
                                         const hipComplexWrapper& c) noexcept {
  return {cuCfmaf(a.v, b.v, c.v)};
}

__host__ __device__ inline hipComplexWrapper operator-(const hipComplexWrapper& a) noexcept {
  return {make_cuComplex(-a.v.x, -a.v.y)};
}

__host__ __device__ inline bool operator==(const hipComplexWrapper& a, const hipComplexWrapper& b) noexcept {
  return a.v.x == b.v.x && a.v.y == b.v.y;
}

__host__ __device__ inline bool operator!=(const hipComplexWrapper& a, const hipComplexWrapper& b) noexcept {
  return !(operator==)(a, b);
}

__host__ __device__ inline hipComplexWrapper operator+(const hipComplexWrapper& a, const hipComplexWrapper& b) noexcept {
  return {cuCaddf(a.v, b.v)};
}

__host__ __device__ inline hipComplexWrapper operator-(const hipComplexWrapper& a, const hipComplexWrapper& b) noexcept {
  return {cuCsubf(a.v, b.v)};
}

__host__ __device__ inline hipComplexWrapper operator*(const hipComplexWrapper& a, const hipComplexWrapper& b) noexcept {
  return {cuCmulf(a.v, b.v)};
}

__host__ __device__ inline hipComplexWrapper operator/(const hipComplexWrapper& a, const hipComplexWrapper& b) noexcept {
  return {cuCdivf(a.v, b.v)};
}

__host__ __device__ inline hipComplexWrapper operator*(const float& a, const hipComplexWrapper& b) noexcept {
  return {make_cuComplex(a * b.v.x, a * b.v.y)};
}

__host__ __device__ inline hipComplexWrapper operator*(const hipComplexWrapper& a, const float& b) noexcept {
  return (operator*)(b, a);
}

__host__ __device__ inline hipComplexWrapper operator/(const hipComplexWrapper& a, const float& b) noexcept {
  return {make_cuComplex(a.v.x / b, a.v.y / b)};
}
#endif

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
#ifdef DLAF_WITH_CUDA
__host__ __device__ inline cuDoubleComplex conj(const cuDoubleComplex& a) noexcept {
  return cuConj(a);
}

__host__ __device__ inline double real(const cuDoubleComplex& a) noexcept {
  return a.x;
}

__host__ __device__ inline double imag(const cuDoubleComplex& a) noexcept {
  return a.y;
}

__host__ __device__ inline cuDoubleComplex fma(const cuDoubleComplex& a, const cuDoubleComplex& b,
                                               const cuDoubleComplex& c) noexcept {
  return cuCfma(a, b, c);
}

__host__ __device__ inline cuDoubleComplex operator-(const cuDoubleComplex& a) noexcept {
  return make_cuDoubleComplex(-a.x, -a.y);
}

__host__ __device__ inline bool operator==(const cuDoubleComplex& a, const cuDoubleComplex& b) noexcept {
  return a.x == b.x && a.y == b.y;
}

__host__ __device__ inline bool operator!=(const cuDoubleComplex& a, const cuDoubleComplex& b) noexcept {
  return !(operator==)(a, b);
}

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

__host__ __device__ inline cuDoubleComplex operator*(const double& a,
                                                     const cuDoubleComplex& b) noexcept {
  return make_cuDoubleComplex(a * b.x, a * b.y);
}

__host__ __device__ inline cuDoubleComplex operator*(const cuDoubleComplex& a,
                                                     const double& b) noexcept {
  return (operator*)(b, a);
}

__host__ __device__ inline cuDoubleComplex operator/(const cuDoubleComplex& a,
                                                     const double& b) noexcept {
  return make_cuDoubleComplex(a.x / b, a.y / b);
}
#elif defined(DLAF_WITH_HIP)
__host__ __device__ inline hipDoubleComplexWrapper conj(const hipDoubleComplexWrapper& a) noexcept {
  return {cuConj(a.v)};
}

__host__ __device__ inline double real(const hipDoubleComplexWrapper& a) noexcept {
  return a.v.x;
}

__host__ __device__ inline double imag(const hipDoubleComplexWrapper& a) noexcept {
  return a.v.y;
}

__host__ __device__ inline hipDoubleComplexWrapper fma(const hipDoubleComplexWrapper& a, const hipDoubleComplexWrapper& b,
                                               const hipDoubleComplexWrapper& c) noexcept {
  return {cuCfma(a.v, b.v, c.v)};
}

__host__ __device__ inline hipDoubleComplexWrapper operator-(const hipDoubleComplexWrapper& a) noexcept {
  return {make_cuDoubleComplex(-a.v.x, -a.v.y)};
}

__host__ __device__ inline bool operator==(const hipDoubleComplexWrapper& a, const hipDoubleComplexWrapper& b) noexcept {
  return a.v.x == b.v.x && a.v.y == b.v.y;
}

__host__ __device__ inline bool operator!=(const hipDoubleComplexWrapper& a, const hipDoubleComplexWrapper& b) noexcept {
  return !(operator==)(a, b);
}

__host__ __device__ inline hipDoubleComplexWrapper operator+(const hipDoubleComplexWrapper& a,
                                                     const hipDoubleComplexWrapper& b) noexcept {
  return {cuCadd(a.v, b.v)};
}

__host__ __device__ inline hipDoubleComplexWrapper operator-(const hipDoubleComplexWrapper& a,
                                                     const hipDoubleComplexWrapper& b) noexcept {
  return {cuCsub(a.v, b.v)};
}

__host__ __device__ inline hipDoubleComplexWrapper operator*(const hipDoubleComplexWrapper& a,
                                                     const hipDoubleComplexWrapper& b) noexcept {
  return {cuCmul(a.v, b.v)};
}

__host__ __device__ inline hipDoubleComplexWrapper operator/(const hipDoubleComplexWrapper& a,
                                                     const hipDoubleComplexWrapper& b) noexcept {
  return {cuCdiv(a.v, b.v)};
}

__host__ __device__ inline hipDoubleComplexWrapper operator*(const double& a,
                                                     const hipDoubleComplexWrapper& b) noexcept {
  return {make_cuDoubleComplex(a * b.v.x, a * b.v.y)};
}

__host__ __device__ inline hipDoubleComplexWrapper operator*(const hipDoubleComplexWrapper& a,
                                                     const double& b) noexcept {
  return (operator*)(b, a);
}

__host__ __device__ inline hipDoubleComplexWrapper operator/(const hipDoubleComplexWrapper& a,
                                                     const double& b) noexcept {
  return {make_cuDoubleComplex(a.v.x / b, a.v.y / b)};
}
#endif

}
}

#endif
