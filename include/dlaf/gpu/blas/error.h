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

#include <exception>
#include <iostream>

#include "dlaf/common/source_location.h"
#include "dlaf/gpu/api.h"
#include "dlaf/gpu/blas/api.h"

namespace dlaf {
namespace internal {

#ifdef DLAF_WITH_GPU

/// CUBLAS equivalent to `cudaGetErrorString()`
/// Reference: https://docs.nvidia.com/cuda/cublas/index.html#cublasstatus_t
inline std::string cublasGetErrorString(cublasStatus_t st) {
  // clang-format off
  switch (st) {
    case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
#if defined(DLAF_WITH_CUDA)
    case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
#else
    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:    return "CUBLAS_STATUS_HANDLE_IS_NULLPTR";
    case HIPBLAS_STATUS_INVALID_ENUM:    return "CUBLAS_STATUS_INVALID_ENUM";
    case HIPBLAS_STATUS_UNKNOWN:    return "CUBLAS_STATUS_UNKNOWN";
#endif
  }
  // clang-format on
  return "UNKNOWN";
}

inline void cublasCall(cublasStatus_t st, const dlaf::common::internal::source_location& info) noexcept {
  if (st != CUBLAS_STATUS_SUCCESS) {
    std::cout << "[CUBLAS ERROR] " << info << std::endl << cublasGetErrorString(st) << std::endl;
    std::terminate();
  }
}

#define DLAF_CUBLAS_CHECK_ERROR(cublas_err) dlaf::internal::cublasCall((cublas_err), SOURCE_LOCATION())

#endif

}
}
