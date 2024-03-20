//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <whip.hpp>

#include <dlaf/eigensolver/reduction_to_trid/gpu/kernels.h>
#include <dlaf/gpu/blas/api.h>
#include <dlaf/util_cuda.h>
#include <dlaf/util_math.h>

#include "dlaf/types.h"

namespace dlaf::eigensolver::internal::gpu {

template <class T>
__global__ void updatePointers(const std::size_t n, T** a, const int ld) {
  if (threadIdx.x < n)
    a[threadIdx.x] += ld;
}

template <class T>
void updatePointers(const std::size_t n, T** b, const SizeType ld, whip::stream_t stream) {
  dim3 nr_threads(to_sizet(n), 1);
  dim3 nr_blocks(1, 1);
  updatePointers<<<nr_blocks, nr_threads, 0, stream>>>(util::cppToCudaCast(n), util::cppToCudaCast(b),
                                                       util::cppToCudaCast(ld));
}

DLAF_GPU_UPDATE_POINTERS_ETI(, float);
DLAF_GPU_UPDATE_POINTERS_ETI(, double);
DLAF_GPU_UPDATE_POINTERS_ETI(, std::complex<float>);
DLAF_GPU_UPDATE_POINTERS_ETI(, std::complex<double>);

}
