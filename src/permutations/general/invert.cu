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

#include "dlaf/permutations/general/invert.h"

constexpr unsigned invert_index_kernel_sz = 256;

__global__ void invertIndexOnDevice(SizeType len, const SizeType* in, SizeType* out) {
  const SizeType i = blockIdx.x * invert_index_kernel_sz + threadIdx.x;
  if (i >= len)
    return;

  out[in[i]] = i;
}

void invertIndexOnDevice(SizeType len, const SizeType* in, SizeType* out, cudaStream_t stream) {
  dim3 nr_threads(invert_index_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(to_sizet(len), to_sizet(invert_index_kernel_sz)));
  invertIndexOnDevice<<<nr_blocks, nr_threads, 0, stream>>>(len, in, out);
}
