//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/util_cuda.h"

#include <complex>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "dlaf/cuda/error.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::util;
using namespace dlaf::test;
using namespace testing;

using TestTypes = ::testing::Types<float, double>;

template <class T>
class CudaUtilTestHost : public ::testing::Test {};
TYPED_TEST_SUITE(CudaUtilTestHost, TestTypes);

template <class T>
class CudaUtilTestDevice : public ::testing::Test {};
TYPED_TEST_SUITE(CudaUtilTestDevice, TestTypes);

template <class T>
struct cudaComplex;

template <>
struct cudaComplex<float> {
  using Type = cuComplex;
};

template <>
struct cudaComplex<double> {
  using Type = cuDoubleComplex;
};

template <class T>
using cudaComplex_t = typename cudaComplex<T>::Type;

TYPED_TEST(CudaUtilTestHost, CppToCudaCastReal) {
  TypeParam x = 3.55f;

  auto val = cppToCudaCast(x);

  EXPECT_TRUE((std::is_same_v<TypeParam, decltype(val)>) );
  EXPECT_EQ(x, val);

  auto ptr = cppToCudaCast(&x);

  EXPECT_TRUE((std::is_same_v<TypeParam*, decltype(ptr)>) );
  EXPECT_EQ(&x, ptr);
}

TYPED_TEST(CudaUtilTestHost, CppToCudaCastComplex) {
  using T = TypeParam;

  T x = 3.55f;
  T y = -2.35f;

  std::complex<T> z(x, y);

  auto val = cppToCudaCast(z);

  EXPECT_TRUE((std::is_same_v<cudaComplex_t<T>, decltype(val)>) );
  EXPECT_EQ(z.real(), val.x);
  EXPECT_EQ(z.imag(), val.y);

  auto ptr = cppToCudaCast(&z);

  EXPECT_TRUE((std::is_same_v<cudaComplex_t<T>*, decltype(ptr)>) );
  EXPECT_EQ(reinterpret_cast<void*>(&z), reinterpret_cast<void*>(ptr));
  EXPECT_EQ(reinterpret_cast<T*>(&z), &(ptr->x));
  EXPECT_EQ(reinterpret_cast<T*>(&z) + 1, &(ptr->y));
}

TYPED_TEST(CudaUtilTestHost, CudaOperatorsReal) {
  using namespace cuda_operators;
  using T = TypeParam;

  const T a = 3.55f;
  const T b = 2.15f;
  const T c = -7.65f;

  EXPECT_EQ(a, conj(a));
  EXPECT_EQ(b, real(b));
  EXPECT_EQ(T{0.f}, imag(c));

  EXPECT_NEAR(a * b + c, cuda_operators::fma(a, b, c), 5 * TypeUtilities<T>::error);
}

#define EXPECT_EQ_COMPLEX(real, imag, val) \
  do {                                     \
    EXPECT_EQ(real, (val).x);              \
    EXPECT_EQ(imag, (val).y);              \
  } while (0)

#define EXPECT_NEAR_COMPLEX(real, imag, val, error) \
  do {                                              \
    EXPECT_NEAR(real, (val).x, error);              \
    EXPECT_NEAR(imag, (val).y, error);              \
  } while (0)

TYPED_TEST(CudaUtilTestHost, CudaOperatorsComplex) {
  using namespace cuda_operators;
  using T = TypeParam;
  using ComplexT = cudaComplex_t<T>;

  const ComplexT a = cppToCudaCast(std::complex<T>(3.55f, -2.35f));
  const ComplexT b = cppToCudaCast(std::complex<T>(2.15f, 0.66f));
  const ComplexT c = cppToCudaCast(std::complex<T>(-7.65f, -5.12f));
  const T d = 7.77;

  EXPECT_EQ_COMPLEX(-(a.x), -(a.y), -a);
  EXPECT_EQ_COMPLEX(a.x, -(a.y), conj(a));
  EXPECT_EQ(b.x, real(b));
  EXPECT_EQ(c.y, imag(c));

  EXPECT_NEAR_COMPLEX(a.x + b.x, a.y + b.y, a + b, 5 * TypeUtilities<T>::error);
  EXPECT_NEAR_COMPLEX(a.x - b.x, a.y - b.y, a - b, 5 * TypeUtilities<T>::error);

  EXPECT_NEAR_COMPLEX(a.x * c.x - a.y * c.y, a.y * c.x + a.x * c.y, a * c, 5 * TypeUtilities<T>::error);
  const T c2 = c.x * c.x + c.y * c.y;
  EXPECT_NEAR_COMPLEX((a.x * c.x + a.y * c.y) / c2, (a.y * c.x - a.x * c.y) / c2, a / c,
                      5 * TypeUtilities<T>::error);

  EXPECT_NEAR_COMPLEX((a * b + c).x, (a * b + c).y, fma(a, b, c), 5 * TypeUtilities<T>::error);

  EXPECT_NEAR_COMPLEX(d * a.x, d * a.y, d * a, 5 * TypeUtilities<T>::error);
  EXPECT_NEAR_COMPLEX(b.x * d, b.y * d, b * d, 5 * TypeUtilities<T>::error);
  EXPECT_NEAR_COMPLEX(c.x / d, c.y / d, c / d, 5 * TypeUtilities<T>::error);
}

template <class T>
__global__ void testOperatorsReal(T a, T b, T c, T* result) {
  using namespace cuda_operators;
  result[0] = conj(a);
  result[1] = real(b);
  result[2] = imag(c);

  result[3] = cuda_operators::fma(a, b, c);
}

TYPED_TEST(CudaUtilTestDevice, CudaOperatorsReal) {
  using T = TypeParam;

  const T a = 3.55f;
  const T b = 2.15f;
  const T c = -7.65f;

  constexpr unsigned res_size = 4;

  T* res_d;
  DLAF_CUDA_CALL(cudaMalloc(&res_d, res_size * sizeof(T)));

  testOperatorsReal<<<1, 1>>>(a, b, c, res_d);

  T* res_h;
  DLAF_CUDA_CALL(cudaMallocHost(&res_h, res_size * sizeof(T)));
  DLAF_CUDA_CALL(cudaMemcpy(res_h, res_d, res_size * sizeof(T), cudaMemcpyDefault));

  EXPECT_EQ(a, res_h[0]);
  EXPECT_EQ(b, res_h[1]);
  EXPECT_EQ(T{0.f}, res_h[2]);

  EXPECT_NEAR(a * b + c, res_h[3], 5 * TypeUtilities<T>::error);

  DLAF_CUDA_CALL(cudaFreeHost(res_h));
  DLAF_CUDA_CALL(cudaFree(res_d));
}

template <class T>
__global__ void testOperatorsComplex(cudaComplex_t<T> a, cudaComplex_t<T> b, cudaComplex_t<T> c, T d,
                                     cudaComplex_t<T>* result, T* result_real) {
  using namespace cuda_operators;
  result[0] = -a;
  result[1] = conj(a);
  result_real[0] = real(b);
  result_real[1] = imag(c);

  result[2] = a + b;
  result[3] = a - b;
  result[4] = a * c;
  result[5] = a / c;

  result[6] = fma(a, b, c);

  result[7] = d * a;
  result[8] = b * d;
  result[9] = c / d;
}

TYPED_TEST(CudaUtilTestDevice, CudaOperatorsComplex) {
  using namespace cuda_operators;
  using T = TypeParam;
  using ComplexT = cudaComplex_t<T>;

  const ComplexT a = cppToCudaCast(std::complex<T>(3.55f, -2.35f));
  const ComplexT b = cppToCudaCast(std::complex<T>(2.15f, 0.66f));
  const ComplexT c = cppToCudaCast(std::complex<T>(-7.65f, -5.12f));
  const T d = 7.77;

  constexpr unsigned res_size = 10;
  constexpr unsigned res_real_size = 2;

  ComplexT* res_d;
  DLAF_CUDA_CALL(cudaMalloc(&res_d, res_size * sizeof(ComplexT)));
  T* res_real_d;
  DLAF_CUDA_CALL(cudaMalloc(&res_real_d, res_real_size * sizeof(T)));

  testOperatorsComplex<<<1, 1>>>(a, b, c, d, res_d, res_real_d);

  ComplexT* res_h;
  DLAF_CUDA_CALL(cudaMallocHost(&res_h, res_size * sizeof(ComplexT)));
  DLAF_CUDA_CALL(cudaMemcpy(res_h, res_d, res_size * sizeof(ComplexT), cudaMemcpyDefault));
  T* res_real_h;
  DLAF_CUDA_CALL(cudaMallocHost(&res_real_h, res_real_size * sizeof(T)));
  DLAF_CUDA_CALL(cudaMemcpy(res_real_h, res_real_d, res_real_size * sizeof(T), cudaMemcpyDefault));

  EXPECT_EQ_COMPLEX(-(a.x), -(a.y), res_h[0]);
  EXPECT_EQ_COMPLEX(a.x, -(a.y), res_h[1]);
  EXPECT_EQ(b.x, res_real_h[0]);
  EXPECT_EQ(c.y, res_real_h[1]);

  EXPECT_NEAR_COMPLEX(a.x + b.x, a.y + b.y, res_h[2], 5 * TypeUtilities<T>::error);
  EXPECT_NEAR_COMPLEX(a.x - b.x, a.y - b.y, res_h[3], 5 * TypeUtilities<T>::error);

  EXPECT_NEAR_COMPLEX(a.x * c.x - a.y * c.y, a.y * c.x + a.x * c.y, res_h[4],
                      5 * TypeUtilities<T>::error);
  const T c2 = c.x * c.x + c.y * c.y;
  EXPECT_NEAR_COMPLEX((a.x * c.x + a.y * c.y) / c2, (a.y * c.x - a.x * c.y) / c2, res_h[5],
                      5 * TypeUtilities<T>::error);

  EXPECT_NEAR_COMPLEX((a * b + c).x, (a * b + c).y, res_h[6], 5 * TypeUtilities<T>::error);

  EXPECT_NEAR_COMPLEX(d * a.x, d * a.y, res_h[7], 5 * TypeUtilities<T>::error);
  EXPECT_NEAR_COMPLEX(b.x * d, b.y * d, res_h[8], 5 * TypeUtilities<T>::error);
  EXPECT_NEAR_COMPLEX(c.x / d, c.y / d, res_h[9], 5 * TypeUtilities<T>::error);

  DLAF_CUDA_CALL(cudaFreeHost(res_real_h));
  DLAF_CUDA_CALL(cudaFreeHost(res_h));
  DLAF_CUDA_CALL(cudaFree(res_real_d));
  DLAF_CUDA_CALL(cudaFree(res_d));
}
