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
#include "dlaf/gpu/rocblas/error.h"

namespace dlaf::gpublas::internal {

#ifdef DLAF_WITH_GPU

inline void checkError(gpublasStatus_t st,
                       const dlaf::common::internal::source_location& info) noexcept {
  if (st != GPUBLAS_STATUS_SUCCESS) {
#ifdef DLAF_WITH_CUDA
    std::cout << "[CUBLAS ERROR] " << info << std::endl
              << cublas::internal::getErrorString(st) << std::endl;
#elif defined(DLAF_WITH_HIP)
    std::cout << "[ROCBLAS ERROR] " << info << std::endl
              << rocblas::internal::getErrorString(st) << std::endl;
#endif
    std::terminate();
  }
}

#define DLAF_GPUBLAS_CHECK_ERROR(gpublas_err) \
  ::dlaf::gpublas::internal::checkError((gpublas_err), SOURCE_LOCATION())

#endif

#ifdef DLAF_WITH_HIP
inline rocblas_status acceptWorkspaceStatus(rocblas_status error) {
  switch (error) {
    case rocblas_status_continue:
    case rocblas_status_size_unchanged:
    case rocblas_status_size_increased:
    case rocblas_status_success:
      return rocblas_status_success;
    default:
      return error;
  }
}

#define DLAF_ROCBLAS_WORKSPACE_CHECK_ERROR(gpublas_err) \
  ::dlaf::gpublas::internal::checkError(acceptWorkspaceStatus(gpublas_err), SOURCE_LOCATION())

#endif

}
