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

#include <blas.hh>
#include "dlaf/gpu/blas/api.h"

namespace dlaf::util {
namespace internal {

template <typename T>
struct BlasToRocblasType {
  using type = T;
};

template <>
struct BlasToRocblasType<std::complex<float>> {
  using type = rocblas_float_complex;
};

template <>
struct BlasToRocblasType<std::complex<double>> {
  using type = rocblas_double_complex;
};

template <typename T>
struct BlasToRocblasType<const T> {
  using type = const typename BlasToRocblasType<T>::type;
};

template <typename T>
struct BlasToRocblasType<T*> {
  using type = typename BlasToRocblasType<T>::type*;
};

}

template <typename T>
constexpr typename internal::BlasToRocblasType<T>::type blasToRocblasCast(T p) {
  return reinterpret_cast<typename internal::BlasToRocblasType<T>::type>(p);
}

inline constexpr rocblas_side blasToRocblas(const blas::Side side) {
  switch (side) {
    case blas::Side::Left:
      return rocblas_side_left;
    case blas::Side::Right:
      return rocblas_side_right;
    default:
      return {};
  }
}

inline constexpr rocblas_fill blasToRocblas(const blas::Uplo uplo) {
  switch (uplo) {
    case blas::Uplo::Lower:
      return rocblas_fill_lower;
    case blas::Uplo::Upper:
      return rocblas_fill_upper;
    case blas::Uplo::General:
      return rocblas_fill_full;
    default:
      return {};
  }
}

inline constexpr rocblas_operation blasToRocblas(const blas::Op op) {
  switch (op) {
    case blas::Op::NoTrans:
      return rocblas_operation_none;
    case blas::Op::Trans:
      return rocblas_operation_transpose;
    case blas::Op::ConjTrans:
      return rocblas_operation_conjugate_transpose;
    default:
      return {};
  }
}

inline constexpr rocblas_diagonal blasToRocblas(const blas::Diag diag) {
  switch (diag) {
    case blas::Diag::Unit:
      return rocblas_diagonal_unit;
    case blas::Diag::NonUnit:
      return rocblas_diagonal_non_unit;
    default:
      return {};
  }
}

inline constexpr rocblas_eform blasToRocblas(const int eigenvalueType) {
  switch (eigenvalueType) {
    case 1:
      return rocblas_eform_ax;
    case 2:
      return rocblas_eform_abx;
    case 3:
      return rocblas_eform_bax;
    default:
      return {};
  }
}

}

#endif
