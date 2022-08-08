//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/eigensolver/tridiag_solver/misc_gpu_kernels.h"
#include "dlaf/gpu/api.h"
#include "dlaf/util_cuda.h"
#include "dlaf/util_math.h"

#include "thrust/execution_policy.h"
#include "thrust/merge.h"

namespace dlaf::eigensolver::internal {

template <class T>
void mergeIndicesOnDevice(const SizeType* begin_ptr, const SizeType* split_ptr, const SizeType* end_ptr,
                          SizeType* out_ptr, const T* v_ptr) {
  auto cmp = [v_ptr] __device__(const SizeType& i1, const SizeType& i2) {
    return v_ptr[i1] < v_ptr[i2];
  };
  // TODO: with Thrust > 1.16 use `thrust::cuda::par_nosync.on(args...)` instead of `thrust::device`
  thrust::merge(thrust::device, begin_ptr, split_ptr, split_ptr, end_ptr, out_ptr, std::move(cmp));
}

DLAF_CUDA_MERGE_INDICES_ETI(, float);
DLAF_CUDA_MERGE_INDICES_ETI(, double);

constexpr unsigned apply_index_sz = 64;

template <class T>
__global__ void applyIndexOnDevice(const SizeType* index_arr, const T* in_arr, T* out_arr) {
  const unsigned i = blockIdx.x * apply_index_sz + threadIdx.x;
  out_arr[i] = in_arr[index_arr[i]];
}

template <class T>
void applyIndexOnDevice(SizeType len, const SizeType* index, const T* in, T* out, cudaStream_t stream) {
  const unsigned ulen = to_uint(len);

  dim3 nr_threads(apply_index_sz);
  dim3 nr_blocks(util::ceilDiv(ulen, apply_index_sz));
  applyIndexOnDevice<<<nr_blocks, nr_threads, 0, stream>>>(index, util::cppToCudaCast(in),
                                                           util::cppToCudaCast(out));
}

DLAF_CUDA_APPLY_INDEX_ETI(, float);
DLAF_CUDA_APPLY_INDEX_ETI(, double);

}
