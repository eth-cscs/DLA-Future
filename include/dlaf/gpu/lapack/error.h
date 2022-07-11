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
#include "dlaf/gpu/lapack/api.h"

#include "dlaf/gpu/cusolver/error.h"
#include "dlaf/gpu/rocsolver/error.h"

#ifdef DLAF_WITH_GPU

namespace dlaf::gpulapack::internal {

inline void checkError(gpulapackStatus_t st,
                       const dlaf::common::internal::source_location& info) noexcept {
#ifdef DLAF_WITH_CUDA
  if (st != CUSOLVER_STATUS_SUCCESS) {
    std::cout << "[CUSOLVER ERROR] " << info << std::endl
              << cusolver::internal::getErrorString(st) << std::endl;
    std::terminate();
  }
#elif defined(DLAF_WITH_HIP)
  if (st != rocblas_status_success && st != rocblas_status_size_unchanged &&
      st != rocblas_status_size_increased && st != rocblas_status_continue) {
    std::cout << "[ROCSOLVER ERROR] " << info << std::endl
              << rocsolver::internal::getErrorString(st) << std::endl;
    std::terminate();
  }
#endif
}

#define DLAF_GPULAPACK_CHECK_ERROR(cusolver_err) \
  ::dlaf::gpulapack::internal::checkError((cusolver_err), SOURCE_LOCATION())

}

#endif
