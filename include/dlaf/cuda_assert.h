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

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace dlaf {

#ifdef DLAF_WITH_CUDA
inline void cuda_assert(cudaError_t err, char const* file, int line) noexcept {
  if (err != cudaSuccess) {
    printf("[CUDA ERROR] %s:%d: '%s'\n", file, line, cudaGetErrorString(err));
    std::abort();
  }
}

#define DLAF_CUDA_ASSERT(cuda_f) cuda_assert((cuda_f), __FILE__, __LINE__)

#endif

}
