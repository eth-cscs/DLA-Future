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
#include <rocsolver/rocsolver.h>

// Use of rocblas when the corresponding rocsolver functions are deprecated
#define cusolverDnChegst    rocsolver_chegst
#define cusolverDnCreate    rocblas_create_handle
#define cusolverDnDestroy   rocblas_destroy_handle
#define cusolverDnDsygst    rocsolver_dsygst
#define cusolverDnGetStream rocblas_get_stream
#define cusolverDnHandle_t  rocblas_handle
#define cusolverDnSetStream rocblas_set_stream
#define cusolverDnSsygst    rocsolver_ssygst
#define cusolverDnZhegst    rocsolver_zhegst

namespace dlaf {

// Error handling
using gpulapackStatus_t = rocblas_status;
constexpr gpulapackStatus_t GPULAPACK_STATUS_SUCCESS = rocblas_status_success;

}

#elif defined(DLAF_WITH_CUDA)

#include <cusolverDn.h>

namespace dlaf {

// Error handling
using gpulapackStatus_t = cusolverStatus_t;
constexpr gpulapackStatus_t GPULAPACK_STATUS_SUCCESS = CUSOLVER_STATUS_SUCCESS;

}

#endif
