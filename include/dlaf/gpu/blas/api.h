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

/// @file

#if defined(DLAF_WITH_GPU)

#define DLAF_DECLARE_GPUBLAS_OP(Name) \
  template <typename T>               \
  struct Name

#endif

#if defined(DLAF_WITH_HIP)

#include <hipblas.h>

#define cublasCall           hipblasCall
#define cublasStatus_t       hipblasStatus_t
#define cublasCherk          hipblasCherk
#define cublasCreate         hipblasCreate
#define cublasCtrsm          hipblasCtrsm
#define cublasDaxpy          hipblasDaxpy
#define cublasDestroy        hipblasDestroy
#define cublasDgemm          hipblasDgemm
#define cublasDiagType_t     hipblasDiagType_t
#define cublasDtrsm          hipblasDtrsm
#define cublasFillMode_t     hipblasFillMode_t
#define cublasGetErrorString hipblasGetErrorString
#define cublasHandle_t       hipblasHandle_t
#define cublasOperation_t    hipblasOperation_t
#define cublasPointerMode_t  hipblasPointerMode_t
#define cublasSetPointerMode hipblasSetPointerMode
#define cublasSetStream      hipblasSetStream
#define cublasSgemm          hipblasSgemm
#define cublasSideMode_t     hipblasSideMode_t
#define cublasStrsm          hipblasStrsm
#define cublasZgemm          hipblasZgemm
#define cublasZherk          hipblasZherk
#define cublasZtrsm          hipblasZtrsm
#define cuComplex            hipblasComplex
#define cuDoubleComplex      hipblasDoubleComplex

#define CUBLAS_DIAG_NON_UNIT           HIPBLAS_DIAG_NON_UNIT
#define CUBLAS_DIAG_UNIT               HIPBLAS_DIAG_UNIT
#define CUBLAS_FILL_MODE_FULL          HIPBLAS_FILL_MODE_FULL
#define CUBLAS_FILL_MODE_LOWER         HIPBLAS_FILL_MODE_LOWER
#define CUBLAS_FILL_MODE_UPPER         HIPBLAS_FILL_MODE_UPPER
#define CUBLAS_OP_C                    HIPBLAS_OP_C
#define CUBLAS_OP_N                    HIPBLAS_OP_N
#define CUBLAS_OP_T                    HIPBLAS_OP_T
#define CUBLAS_POINTER_MODE_HOST       HIPBLAS_POINTER_MODE_HOST
#define CUBLAS_SIDE_LEFT               HIPBLAS_SIDE_LEFT
#define CUBLAS_SIDE_RIGHT              HIPBLAS_SIDE_RIGHT
#define CUBLAS_STATUS_ALLOC_FAILED     HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_ARCH_MISMATCH    HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR   HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_INVALID_VALUE    HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_MAPPING_ERROR    HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_NOT_INITIALIZED  HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_NOT_SUPPORTED    HIPBLAS_STATUS_NOT_SUPPORTED
#define CUBLAS_STATUS_SUCCESS          HIPBLAS_STATUS_SUCCESS

// Need this macro before the cublasCall definition in error.h
// TODO: This is dirty. This kind of include structure should not be
// necessary... split headers if necessary.
#include "dlaf/gpu/blas/error.h"

#define DLAF_DEFINE_GPUBLAS_OP(Name, Type, f)                           \
  template <>                                                           \
  struct Name<Type> {                                                   \
    template <typename... Args>                                         \
    static void call(Args&&... args) {                                  \
      DLAF_CUBLAS_CHECK_ERROR(hipblas##f(std::forward<Args>(args)...)); \
    }                                                                   \
  }

#elif defined(DLAF_WITH_CUDA)

#include <cuComplex.h>
#include <cublas_v2.h>
#include "dlaf/gpu/blas/error.h"

#define DLAF_DEFINE_GPUBLAS_OP(Name, Type, f)                               \
  template <>                                                               \
  struct Name<Type> {                                                       \
    template <typename... Args>                                             \
    static void call(Args&&... args) {                                      \
      DLAF_CUBLAS_CHECK_ERROR(cublas##f##_v2(std::forward<Args>(args)...)); \
    }                                                                       \
  }
#endif

#if defined(DLAF_WITH_GPU)

#define DLAF_MAKE_GPUBLAS_OP(Name, f)                      \
  DLAF_DECLARE_GPUBLAS_OP(Name);                           \
  DLAF_DEFINE_GPUBLAS_OP(Name, float, S##f);               \
  DLAF_DEFINE_GPUBLAS_OP(Name, double, D##f);              \
  DLAF_DEFINE_GPUBLAS_OP(Name, std::complex<float>, C##f); \
  DLAF_DEFINE_GPUBLAS_OP(Name, std::complex<double>, Z##f)

#define DLAF_MAKE_GPUBLAS_SYHE_OP(Name, f)                   \
  DLAF_DECLARE_GPUBLAS_OP(Name);                             \
  DLAF_DEFINE_GPUBLAS_OP(Name, float, Ssy##f);               \
  DLAF_DEFINE_GPUBLAS_OP(Name, double, Dsy##f);              \
  DLAF_DEFINE_GPUBLAS_OP(Name, std::complex<float>, Che##f); \
  DLAF_DEFINE_GPUBLAS_OP(Name, std::complex<double>, Zhe##f)

namespace dlaf::gpublas {

// Level 1
DLAF_MAKE_GPUBLAS_OP(Axpy, axpy);

// Level 2
DLAF_MAKE_GPUBLAS_OP(Gemv, gemv);

DLAF_MAKE_GPUBLAS_OP(Trmv, trmv);

// Level 3
DLAF_MAKE_GPUBLAS_OP(Gemm, gemm);

DLAF_MAKE_GPUBLAS_SYHE_OP(Hemm, mm);

DLAF_MAKE_GPUBLAS_SYHE_OP(Her2k, r2k);

DLAF_MAKE_GPUBLAS_SYHE_OP(Herk, rk);

DLAF_MAKE_GPUBLAS_OP(Trmm, trmm);

DLAF_MAKE_GPUBLAS_OP(Trsm, trsm);

}

#endif
