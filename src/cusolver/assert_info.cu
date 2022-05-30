//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <stdio.h>
#include "dlaf/gpu/lapack/assert_info.h"

#ifdef DLAF_ASSERT_ENABLE

#define DLAF_DEFINE_CUSOLVER_ASSERT_INFO(func)                                                 \
  void assertInfo##func(cudaStream_t stream, int* info) {                                      \
    dlaf::cusolver::assert_info<<<1, 1, 0, stream>>>(info, [] __device__() { return #func; }); \
  }

#else

#define DLAF_DEFINE_CUSOLVER_ASSERT_INFO(func) \
  void assertInfo##func(cudaStream_t stream, int* info) {}

#endif

namespace dlaf {
namespace cusolver {

template <class F>
__global__ void assert_info(int* info, F func) {
  if (*info != 0) {
    printf("Error %s: info != 0 (%d)\n", func(), *info);
#ifdef DLAF_WITH_CUDA
    __trap();
#else
    abort();
#endif
  }
}

DLAF_DEFINE_CUSOLVER_ASSERT_INFO(Potrf)
DLAF_DEFINE_CUSOLVER_ASSERT_INFO(Hegst)

}
}
