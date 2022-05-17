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

#if defined(DLAF_WITH_HIP)

#include <hipblas.h>

#define cublasCall           hipblasCall
#define cublasStatus_t       hipblasStatus_t
#define cublasCherk          hipblasCherk
#define cublasCreate         hipblasCreate
#define cublasCtrmm          hipblasCtrmm
#define cublasCtrsm          hipblasCtrsm
#define cublasDaxpy          hipblasDaxpy
#define cublasDestroy        hipblasDestroy
#define cublasDgemm          hipblasDgemm
#define cublasDiagType_t     hipblasDiagType_t
#define cublasDtrmm          hipblasDtrmm
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
#define cublasStrmm          hipblasStrmm
#define cublasStrsm          hipblasStrsm
#define cublasZgemm          hipblasZgemm
#define cublasZherk          hipblasZherk
#define cublasZtrmm          hipblasZtrmm
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

#elif defined(DLAF_WITH_CUDA)

#include <cuComplex.h>
#include <cublas_v2.h>

#endif
