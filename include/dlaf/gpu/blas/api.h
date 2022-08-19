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

#ifdef DLAF_WITH_HIP

#include <rocblas/rocblas.h>

#define cublasCherks         rocblas_cherk
#define cublasCreate         rocblas_create_handle
#define cublasCtrmm          rocblas_ctrmm
#define cublasCtrsm          rocblas_ctrsm
#define cublasDaxpy          rocblas_daxpy
#define cublasDestroy        rocblas_destroy_handle
#define cublasDgemm          rocblas_dgemm
#define cublasDiagType_t     rocblas_diagonal
#define cublasDtrmm          rocblas_dtrmm
#define cublasDtrsm          rocblas_dtrsm
#define cublasFillMode_t     rocblas_fill
#define cublasGetStream      rocblas_get_stream
#define cublasHandle_t       rocblas_handle
#define cublasOperation_t    rocblas_operation
#define cublasPointerMode_t  rocblas_pointer_mode
#define cublasSetPointerMode rocblas_set_pointer_mode
#define cublasSetStream      rocblas_set_stream
#define cublasSgemm          rocblas_sgemm
#define cublasSideMode_t     rocblas_side
#define cublasStrmm          rocblas_strmm
#define cublasStrsm          rocblas_strsm
#define cublasZgemm          rocblas_zgemm
#define cublasZherk          rocblas_zherk
#define cublasZtrmm          rocblas_ztrmm
#define cublasZtrsm          rocblas_ztrsm
#define cuComplex            hipFloatComplex
#define cuDoubleComplex      hipDoubleComplex

#define CUBLAS_DIAG_NON_UNIT     rocblas_diagonal_non_unit
#define CUBLAS_DIAG_UNIT         rocblas_diagonal_unit
#define CUBLAS_FILL_MODE_FULL    rocblas_fill_full
#define CUBLAS_FILL_MODE_LOWER   rocblas_fill_lower
#define CUBLAS_FILL_MODE_UPPER   rocblas_fill_upper
#define CUBLAS_OP_C              rocblas_operation_conjugate_transpose
#define CUBLAS_OP_N              rocblas_operation_none
#define CUBLAS_OP_T              rocblas_operation_transpose
#define CUBLAS_POINTER_MODE_HOST rocblas_pointer_mode_host
#define CUBLAS_SIDE_LEFT         rocblas_side_left
#define CUBLAS_SIDE_RIGHT        rocblas_side_right

namespace dlaf::gpublas::internal {

// Error handling
using gpublasStatus_t = rocblas_status;
inline constexpr gpublasStatus_t GPUBLAS_STATUS_SUCCESS = rocblas_status_success;

}

#elif defined(DLAF_WITH_CUDA)

#include <cuComplex.h>
#include <cublas_v2.h>

namespace dlaf {

// Error handling
using gpublasStatus_t = cublasStatus_t;
inline constexpr gpublasStatus_t GPUBLAS_STATUS_SUCCESS = CUBLAS_STATUS_SUCCESS;

}

#endif
