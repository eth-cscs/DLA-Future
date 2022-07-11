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
#include "dlaf/gpu/blas/api.h"
#include "dlaf/gpu/cublas/error.h"
#include "dlaf/gpu/hipblas/error.h"

namespace dlaf::gpublas::internal {

#ifdef DLAF_WITH_GPU

inline void checkError(gpublasStatus_t st,
                       const dlaf::common::internal::source_location& info) noexcept {
#ifdef DLAF_WITH_CUDA
  if (st != CUBLAS_STATUS_SUCCESS) {
    std::cout << "[CUBLAS ERROR] " << info << std::endl
              << cublas::internal::getErrorString(st) << std::endl;
    std::terminate();
  }
#elif defined(DLAF_WITH_HIP)
  if (st != HIPBLAS_STATUS_SUCCESS) {
    std::cout << "[HIPBLAS ERROR] " << info << std::endl
              << hipblas::internal::getErrorString(st) << std::endl;
    std::terminate();
  }
#endif
}

#ifdef DLAF_WITH_HIP
// Lifted from hipBLAS/library/src/hcc_detail/hipblas.cpp, with missing
// enumeration values added. This should be removed when hipBLAS is no longer
// used.
inline hipblasStatus_t rocBLASStatusToHIPStatus(rocblas_status error) {
  switch (error) {
    case rocblas_status_continue:
    case rocblas_status_size_unchanged:
    case rocblas_status_size_increased:
    case rocblas_status_success:
      return HIPBLAS_STATUS_SUCCESS;
    case rocblas_status_invalid_handle:
      return HIPBLAS_STATUS_NOT_INITIALIZED;
    case rocblas_status_not_implemented:
      return HIPBLAS_STATUS_NOT_SUPPORTED;
    case rocblas_status_invalid_pointer:
    case rocblas_status_invalid_size:
    case rocblas_status_invalid_value:
      return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblas_status_memory_error:
      return HIPBLAS_STATUS_ALLOC_FAILED;
    case rocblas_status_internal_error:
      return HIPBLAS_STATUS_INTERNAL_ERROR;
    case rocblas_status_perf_degraded:
    case rocblas_status_size_query_mismatch:
    case rocblas_status_check_numerics_fail:
    default:
      return HIPBLAS_STATUS_UNKNOWN;
  }
}

inline void checkError(rocblas_status st, const dlaf::common::internal::source_location& info) noexcept {
  checkError(rocBLASStatusToHIPStatus(st), info);
}
#endif

#define DLAF_GPUBLAS_CHECK_ERROR(gpublas_err) \
  ::dlaf::gpublas::internal::checkError((gpublas_err), SOURCE_LOCATION())

#endif

}
