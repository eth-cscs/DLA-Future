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

namespace dlaf::gpublas::rocblas::internal {

inline std::string getErrorString(rocblas_status st) {
  // clang-format off
  switch (st) {
    case rocblas_status_check_numerics_fail:    return "rocblas_status_check_numerics_fail";
    case rocblas_status_continue:               return "rocblas_status_continue";
    case rocblas_status_internal_error:         return "rocblas_status_internal_error";
    case rocblas_status_invalid_handle:         return "rocblas_status_invalid_handle";
    case rocblas_status_invalid_pointer:        return "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size:           return "rocblas_status_invalid_size";
    case rocblas_status_invalid_value:          return "rocblas_status_invalid_value";
    case rocblas_status_memory_error:           return "rocblas_status_memory_error";
    case rocblas_status_perf_degraded:          return "rocblas_status_perf_degraded";
    case rocblas_status_not_implemented:        return "rocblas_status_not_implemented";
    case rocblas_status_size_increased:         return "rocblas_status_size_increased";
    case rocblas_status_size_query_mismatch:    return "rocblas_status_size_query_mismatch";
    case rocblas_status_size_unchanged:         return "rocblas_status_size_unchanged";
    case rocblas_status_success:                return "rocblas_status_success";
  }
  // clang-format on
  return "UNKNOWN";
}

}
#endif
