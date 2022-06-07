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

#ifdef DLAF_WITH_GPU

namespace dlaf::gpu {

inline void checkError(gpuError_t err, const dlaf::common::internal::source_location& info) noexcept {
#ifdef DLAF_WITH_CUDA
  if (err != cudaSuccess) {
    std::cout << "[CUDA ERROR] " << info << std::endl << cudaGetErrorString(err) << std::endl;
    std::terminate();
  }
#elif defined(DLAF_WITH_HIP)
  if (err != hipSuccess) {
    std::cout << "[HIP ERROR] " << info << std::endl << hipGetErrorString(err) << std::endl;
    std::terminate();
  }
#endif
}

#define DLAF_GPU_CHECK_ERROR(cuda_err) ::dlaf::gpu::checkError((cuda_err), SOURCE_LOCATION())
}

#endif
