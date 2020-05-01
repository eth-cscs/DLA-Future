//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#ifdef DLAF_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <cstdio>
#include <cstdlib>

namespace dlaf {
namespace internal {

#ifdef DLAF_WITH_CUDA

inline void cuda_call(cudaError_t err, char const* file, int line) noexcept {
  if (err != cudaSuccess) {
    std::printf("[CUDA ERROR] %s:%d: '%s'\n", file, line, cudaGetErrorString(err));
    // std::abort();
  }
}

#define DLAF_CUDA_CALL(cuda_f) dlaf::internal::cuda_call((cuda_f), __FILE__, __LINE__)

#endif

}
}
