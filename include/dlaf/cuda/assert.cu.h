//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once
#include <stdio.h>

#ifdef DLAF_ASSERT_HEAVY_ENABLE

#define DLAF_GPU_ASSERT_HEAVY(expr) \
  dlaf::cuda::gpuAssert(expr,       \
                        [] __device__() { printf("GPU assertion failed: %s:%d", __FILE__, __LINE__); })

#else
#define #define DLAF_GPU_ASSERT_HEAVY(expr)
#endif

namespace dlaf {
namespace cuda {

template <class PrintFunc>
__device__ void gpuAssert(bool expr, PrintFunc&& print_func) {
  if (!expr) {
    print_func();
    __trap();
  }
}

}
}
