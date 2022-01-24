//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/cuda/assert.cu.h"
#include "dlaf/factorization/qr/t_factor_kernels.h"
#include "dlaf/util_cuda.h"
#include "dlaf/util_math.h"

namespace dlaf::factorization::internal::tfactor_l {
namespace kernels {

using namespace dlaf::util::cuda_operators;

struct TfactorImplicit1Params {
  static constexpr unsigned kernel_tile_size = 32;
};

template <class T>
__global__ void tfactorImplicit1(const unsigned n, const T* tau, const T* v, const unsigned ldv, T* t,
                                 const unsigned ldt) {
  if (blockIdx.x < blockIdx.y)
    return;

  constexpr unsigned kernel_tile_size = TfactorImplicit1Params::kernel_tile_size;

  DLAF_GPU_ASSERT_HEAVY(kernel_tile_size == blockDim.x);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.z);
  DLAF_GPU_ASSERT_HEAVY(n <= kernel_tile_size * gridDim.x);
  DLAF_GPU_ASSERT_HEAVY(n > kernel_tile_size * (gridDim.x - 1));
  DLAF_GPU_ASSERT_HEAVY(gridDim.y == gridDim.x);
  DLAF_GPU_ASSERT_HEAVY(1 == gridDim.z);

  __shared__ T tile[kernel_tile_size][kernel_tile_size + 1];
  if (blockIdx.x == blockIdx.y) {
    const unsigned i = blockIdx.x * kernel_tile_size + threadIdx.x;
    const unsigned j = blockIdx.y * kernel_tile_size;

    const unsigned k_max = min(kernel_tile_size, n - j);

    if (i < n)
      for (unsigned k = 0; k < k_max; ++k)
        if (j + k == i)
          tile[k][threadIdx.x] = tau[i];
        else if (j + k < i)
          tile[k][threadIdx.x] = -tau[i] * conj(v[i + (j + k) * ldv]);

    __syncthreads();

    // Note: Transposed top left corner index
    //       do not change as blockIdx.x == blockIdx.y

    if (i < n)
      for (unsigned k = 0; k < k_max; ++k)
        if (i <= j + k)
          t[i + (j + k) * ldt] = tile[threadIdx.x][k];
  }

  else {
    {
      const unsigned i = blockIdx.x * kernel_tile_size + threadIdx.x;
      const unsigned j = blockIdx.y * kernel_tile_size;

      if (i < n)
        for (unsigned k = 0; k < kernel_tile_size; ++k)
          tile[k][threadIdx.x] = -tau[i] * conj(v[i + (j + k) * ldv]);
    }

    __syncthreads();

    {
      // Transposed top left corner index
      const unsigned i = blockIdx.y * kernel_tile_size + threadIdx.x;
      const unsigned j = blockIdx.x * kernel_tile_size;

      const unsigned k_max = min(kernel_tile_size, n - j);
      for (unsigned k = 0; k < k_max; ++k)
        t[i + (j + k) * ldt] = tile[threadIdx.x][k];
    }
  }
}
}
#include <iostream>
template <class T>
void tfactorImplicit1(const SizeType n, const T* tau, const T* v, const SizeType ldv, T* t,
                      const SizeType ldt, cudaStream_t stream) {
  constexpr unsigned kernel_tile_size = kernels::TfactorImplicit1Params::kernel_tile_size;
  const unsigned un = to_uint(n);

  dim3 nr_threads(kernel_tile_size);
  dim3 nr_blocks(util::ceilDiv(un, kernel_tile_size), util::ceilDiv(un, kernel_tile_size));
  kernels::tfactorImplicit1<<<nr_blocks, nr_threads, 0, stream>>>(un, util::cppToCudaCast(tau),
                                                                  util::cppToCudaCast(v), to_uint(ldv),
                                                                  util::cppToCudaCast(t), to_uint(ldt));
}

DLAF_FACTORIZATION_TFACTOR_IMPLICIT1_ETI(, float);
DLAF_FACTORIZATION_TFACTOR_IMPLICIT1_ETI(, double);
DLAF_FACTORIZATION_TFACTOR_IMPLICIT1_ETI(, std::complex<float>);
DLAF_FACTORIZATION_TFACTOR_IMPLICIT1_ETI(, std::complex<double>);

}
