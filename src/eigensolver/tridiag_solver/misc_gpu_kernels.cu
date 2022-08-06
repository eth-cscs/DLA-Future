//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/eigensolver/tridiag_solver/misc_gpu_kernels.h>

#include <thrust/execution_policy.h>
#include <thrust/merge.h>


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

}
