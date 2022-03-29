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
#include "dlaf/lapack/gpu/laset.h"
#include "dlaf/util_cublas.h"
#include "dlaf/util_math.h"

namespace dlaf::gpulapack {
namespace kernels {

using namespace dlaf::util::cuda_operators;

struct LasetParams {
  static constexpr unsigned kernel_tile_size_rows = 64;
  static constexpr unsigned kernel_tile_size_cols = 64;
};

template <class T>
__device__ void setAll(const unsigned m, const unsigned n, const T alpha, T* a, const unsigned lda) {
  constexpr unsigned kernel_tile_size_rows = LasetParams::kernel_tile_size_rows;
  constexpr unsigned kernel_tile_size_cols = LasetParams::kernel_tile_size_cols;

  const unsigned i = blockIdx.x * kernel_tile_size_rows + threadIdx.x;
  const unsigned j = blockIdx.y * kernel_tile_size_cols;

  const unsigned k_max = min(j + kernel_tile_size_cols, n);
  if (i < m) {
    for (unsigned k = j; k < k_max; ++k) {
      a[i + k * lda] = alpha;
    }
  }
}

template <bool (*comp)(unsigned, unsigned), class T>
__device__ void setDiag(const unsigned m, const unsigned n, const T alpha, const T beta, T* a,
                        const unsigned lda) {
  constexpr unsigned kernel_tile_size_rows = LasetParams::kernel_tile_size_rows;
  constexpr unsigned kernel_tile_size_cols = LasetParams::kernel_tile_size_cols;

  const unsigned i = blockIdx.x * kernel_tile_size_rows + threadIdx.x;
  const unsigned j = blockIdx.y * kernel_tile_size_cols;

  const unsigned k_max = min(j + kernel_tile_size_cols, n);

  if (i < m) {
    for (unsigned k = j; k < k_max; ++k) {
      if (i == k)
        a[i + k * lda] = beta;
      else if (comp(i, k))
        a[i + k * lda] = alpha;
    }
  }
}

template <class T>
__global__ void laset(cublasFillMode_t uplo, const unsigned m, const unsigned n, const T alpha,
                      const T beta, T* a, const unsigned lda) {
  constexpr unsigned kernel_tile_size_rows = LasetParams::kernel_tile_size_rows;
  constexpr unsigned kernel_tile_size_cols = LasetParams::kernel_tile_size_cols;

  DLAF_GPU_ASSERT_HEAVY(kernel_tile_size_rows % kernel_tile_size_cols == 0);
  DLAF_GPU_ASSERT_HEAVY(kernel_tile_size_rows == blockDim.x);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.z);
  DLAF_GPU_ASSERT_HEAVY(gridDim.x == ceilDiv(m, kernel_tile_size_rows));
  DLAF_GPU_ASSERT_HEAVY(gridDim.y == ceilDiv(n, kernel_tile_size_cols));
  DLAF_GPU_ASSERT_HEAVY(1 == gridDim.z);

  const unsigned i = blockIdx.x;
  const unsigned j = blockIdx.y * kernel_tile_size_cols / kernel_tile_size_rows;
  // if (i == j) the kernel tile contains parts of the diagonal

  if (uplo == CUBLAS_FILL_MODE_LOWER) {
    if (i == j)
      setDiag<dlaf::util::isLower>(m, n, alpha, beta, a, lda);
    else if (i > j)
      setAll(m, n, alpha, a, lda);
  }
  else if (uplo == CUBLAS_FILL_MODE_UPPER) {
    if (i == j)
      setDiag<dlaf::util::isUpper>(m, n, alpha, beta, a, lda);
    else if (i < j)
      setAll(m, n, alpha, a, lda);
  }
  else {
    if (i == j && alpha != beta)
      setDiag<dlaf::util::isGeneral>(m, n, alpha, beta, a, lda);
    else
      setAll(m, n, alpha, a, lda);
  }
}
}

template <class T>
void laset(cublasFillMode_t uplo, SizeType m, SizeType n, T alpha, T beta, T* a, SizeType lda,
           cudaStream_t stream) {
  constexpr unsigned kernel_tile_size_rows = kernels::LasetParams::kernel_tile_size_rows;
  constexpr unsigned kernel_tile_size_cols = kernels::LasetParams::kernel_tile_size_cols;
  const unsigned um = to_uint(m);
  const unsigned un = to_uint(n);

  dim3 nr_threads(kernel_tile_size_rows, 1);
  dim3 nr_blocks(util::ceilDiv(um, kernel_tile_size_rows), util::ceilDiv(un, kernel_tile_size_cols));
  kernels::laset<<<nr_blocks, nr_threads, 0, stream>>>(uplo, um, un, util::cppToCudaCast(alpha),
                                                       util::cppToCudaCast(beta), util::cppToCudaCast(a),
                                                       to_uint(lda));
}

DLAF_CUBLAS_LASET_ETI(, float);
DLAF_CUBLAS_LASET_ETI(, double);
DLAF_CUBLAS_LASET_ETI(, std::complex<float>);
DLAF_CUBLAS_LASET_ETI(, std::complex<double>);
}
