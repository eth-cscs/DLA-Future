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

namespace dlaf::gpublas {

#ifdef DLAF_WITH_GPU

inline void checkError(gpublasStatus_t st,
                       const dlaf::common::internal::source_location& info) noexcept {
#ifdef DLAF_WITH_CUDA
  if (st != CUBLAS_STATUS_SUCCESS) {
    std::cout << "[CUBLAS ERROR] " << info << std::endl << cublas::getErrorString(st) << std::endl;
    std::terminate();
  }
#elif defined(DLAF_WITH_HIP)
  if (st != HIPBLAS_STATUS_SUCCESS) {
    std::cout << "[HIPBLAS ERROR] " << info << std::endl << hipblas::getErrorString(st) << std::endl;
    std::terminate();
  }
#endif
}

#define DLAF_GPUBLAS_CHECK_ERROR(gpublas_err) dlaf::gpublas::checkError((gpublas_err), SOURCE_LOCATION())

#endif

}
