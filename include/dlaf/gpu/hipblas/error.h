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
#include "dlaf/gpu/blas/api.h"

#ifdef DLAF_WITH_HIP

namespace dlaf::gpublas::hipblas::internal {

inline std::string getErrorString(hipblasStatus_t st) {
  // clang-format off
  switch (st) {
    case HIPBLAS_STATUS_SUCCESS:            return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED:    return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED:       return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE:      return "HIPBLAS_STATUS_INVALID_VALUE";
    case HIPBLAS_STATUS_ARCH_MISMATCH:      return "HIPBLAS_STATUS_ARCH_MISMATCH";
    case HIPBLAS_STATUS_MAPPING_ERROR:      return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED:   return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR:     return "HIPBLAS_STATUS_INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED:      return "HIPBLAS_STATUS_NOT_SUPPORTED";
    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:  return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";
    case HIPBLAS_STATUS_INVALID_ENUM:       return "HIPBLAS_STATUS_INVALID_ENUM";
    case HIPBLAS_STATUS_UNKNOWN:            return "HIPBLAS_STATUS_UNKNOWN";
  }
  // clang-format on
  return "UNKNOWN";
}

}
#endif
