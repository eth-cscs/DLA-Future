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

namespace dlaf {
namespace internal {

#ifdef DLAF_WITH_GPU

inline void cudaCall(cudaError_t err, const dlaf::common::internal::source_location& info) noexcept {
  if (err != cudaSuccess) {
    std::cout << "[CUDA ERROR] " << info << std::endl << cudaGetErrorString(err) << std::endl;
    std::terminate();
  }
}

#define DLAF_CUDA_CHECK_ERROR(cuda_err) dlaf::internal::cudaCall((cuda_err), SOURCE_LOCATION())

#endif
}
}
