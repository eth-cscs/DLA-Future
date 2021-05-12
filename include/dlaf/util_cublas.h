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

#ifdef DLAF_WITH_CUDA

#include <cublas_v2.h>
#include <blas.hh>

namespace dlaf {
namespace util {
namespace internal {

template <typename T>
struct BlasToCublasType {
  using type = T;
};

template <>
struct BlasToCublasType<std::complex<float>> {
  using type = cuComplex;
};

template <>
struct BlasToCublasType<std::complex<double>> {
  using type = cuDoubleComplex;
};

template <typename T>
struct BlasToCublasType<const T> {
  using type = const typename BlasToCublasType<T>::type;
};

template <typename T>
struct BlasToCublasType<T*> {
  using type = typename BlasToCublasType<T>::type*;
};

}

template <typename T>
constexpr typename internal::BlasToCublasType<T>::type blasToCublasCast(T p) {
  return reinterpret_cast<typename internal::BlasToCublasType<T>::type>(p);
}

inline constexpr cublasSideMode_t blasToCublas(const blas::Side side) {
  switch (side) {
    case blas::Side::Left:
      return CUBLAS_SIDE_LEFT;
    case blas::Side::Right:
      return CUBLAS_SIDE_RIGHT;
    default:
      return {};
  }
}

inline constexpr cublasFillMode_t blasToCublas(const blas::Uplo uplo) {
  switch (uplo) {
    case blas::Uplo::Lower:
      return CUBLAS_FILL_MODE_LOWER;
    case blas::Uplo::Upper:
      return CUBLAS_FILL_MODE_UPPER;
    case blas::Uplo::General:
      return CUBLAS_FILL_MODE_FULL;
    default:
      return {};
  }
}

inline constexpr cublasOperation_t blasToCublas(const blas::Op op) {
  switch (op) {
    case blas::Op::NoTrans:
      return CUBLAS_OP_N;
    case blas::Op::Trans:
      return CUBLAS_OP_T;
    case blas::Op::ConjTrans:
      return CUBLAS_OP_C;
    default:
      return {};
  }
}

inline constexpr cublasDiagType_t blasToCublas(const blas::Diag diag) {
  switch (diag) {
    case blas::Diag::Unit:
      return CUBLAS_DIAG_UNIT;
    case blas::Diag::NonUnit:
      return CUBLAS_DIAG_NON_UNIT;
    default:
      return {};
  }
}

}
}

#endif
