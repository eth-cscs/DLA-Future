//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/gpu/api.h"
#include "dlaf/gpu/assert.cu.h"
#include "dlaf/gpu/blas/api.h"
#include "dlaf/gpu/error.h"
#include "dlaf/lapack/gpu/lacpy.h"
#include "dlaf/types.h"
#include "dlaf/util_cublas.h"
#include "dlaf/util_math.h"

namespace dlaf::gpulapack {
namespace kernels {

using namespace dlaf::util::cuda_operators;

struct LacpyParams {
  static constexpr unsigned kernel_tile_size_rows = 64;
  static constexpr unsigned kernel_tile_size_cols = 16;
};

template <class T>
__device__ void copyAll(const unsigned m, const unsigned n, const T* a, const unsigned lda, T* b,
                        const unsigned ldb) {
  constexpr unsigned kernel_tile_size_rows = LacpyParams::kernel_tile_size_rows;
  constexpr unsigned kernel_tile_size_cols = LacpyParams::kernel_tile_size_cols;

  const unsigned i = blockIdx.x * kernel_tile_size_rows + threadIdx.x;
  const unsigned j = blockIdx.y * kernel_tile_size_cols;

  if (i >= m)
    return;

  const unsigned k_max = min(j + kernel_tile_size_cols, n);

  for (unsigned k = j; k < k_max; ++k)
    b[i + k * ldb] = a[i + k * lda];
}

template <bool (*comp)(unsigned, unsigned), class T>
__device__ void copyDiag(const unsigned m, const unsigned n, const T* a, const unsigned lda, T* b,
                         const unsigned ldb) {
  constexpr unsigned kernel_tile_size_rows = LacpyParams::kernel_tile_size_rows;
  constexpr unsigned kernel_tile_size_cols = LacpyParams::kernel_tile_size_cols;

  const unsigned i = blockIdx.x * kernel_tile_size_rows + threadIdx.x;
  const unsigned j = blockIdx.y * kernel_tile_size_cols;

  if (i >= m)
    return;

  const unsigned k_max = min(j + kernel_tile_size_cols, n);

  for (unsigned k = j; k < k_max; ++k)
    if (comp(i, k))
      b[i + k * ldb] = a[i + k * lda];
}

template <class T>
__global__ void lacpy(cublasFillMode_t uplo, const unsigned m, const unsigned n, const T* a,
                      const unsigned lda, T* b, const unsigned ldb) {
  constexpr unsigned kernel_tile_size_rows = LacpyParams::kernel_tile_size_rows;
  constexpr unsigned kernel_tile_size_cols = LacpyParams::kernel_tile_size_cols;

  DLAF_GPU_ASSERT_HEAVY(kernel_tile_size_rows % kernel_tile_size_cols == 0);
  DLAF_GPU_ASSERT_HEAVY(kernel_tile_size_rows == blockDim.x);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.z);
  DLAF_GPU_ASSERT_HEAVY(gridDim.x == ceilDiv(m, kernel_tile_size_rows));
  DLAF_GPU_ASSERT_HEAVY(gridDim.y == ceilDiv(n, kernel_tile_size_cols));
  DLAF_GPU_ASSERT_HEAVY(1 == gridDim.z);

  const unsigned i = blockIdx.x;
  const unsigned j = blockIdx.y * kernel_tile_size_cols / kernel_tile_size_rows;

  // Note: if (i == j) the kernel tile contains parts of the diagonal

  switch (uplo) {
    case CUBLAS_FILL_MODE_LOWER:
      if (i == j)
        copyDiag<dlaf::util::isLower>(m, n, a, lda, b, ldb);
      else if (i > j)
        copyAll(m, n, a, lda, b, ldb);
      break;
    case CUBLAS_FILL_MODE_UPPER:
      if (i == j)
        copyDiag<dlaf::util::isUpper>(m, n, a, lda, b, ldb);
      else if (i < j)
        copyAll(m, n, a, lda, b, ldb);
      break;
    case CUBLAS_FILL_MODE_FULL:
      // Note: it is more appropriate to use cudaMemcpy2DAsync in this case
      DLAF_GPU_ASSERT_HEAVY(false);
      // copyAll(m, n, a, lda, b, ldb);
      break;
  }
}
}

template <class T>
void lacpy(const blas::Uplo uplo, const SizeType m, const SizeType n, const T* a, const SizeType lda,
           T* b, const SizeType ldb, const cudaStream_t stream) {
  if (m == 0 || n == 0)
    return;

  DLAF_ASSERT_HEAVY(m <= lda, m, lda);
  DLAF_ASSERT_HEAVY(m <= ldb, m, ldb);

  constexpr unsigned kernel_tile_size_rows = kernels::LacpyParams::kernel_tile_size_rows;
  constexpr unsigned kernel_tile_size_cols = kernels::LacpyParams::kernel_tile_size_cols;

  if (uplo == blas::Uplo::General) {
    const cudaMemcpyKind kind = cudaMemcpyDefault;
    DLAF_GPU_CHECK_ERROR(cudaMemcpy2DAsync(b, to_sizet(ldb) * sizeof(T), a, to_sizet(lda) * sizeof(T),
                                           to_sizet(m) * sizeof(T), to_sizet(n), kind, stream));
  }
  else {
    const unsigned um = to_uint(m);
    const unsigned un = to_uint(n);

    const dim3 nr_threads(kernel_tile_size_rows, 1);
    const dim3 nr_blocks(util::ceilDiv(um, kernel_tile_size_rows),
                         util::ceilDiv(un, kernel_tile_size_cols));
    kernels::lacpy<<<nr_blocks, nr_threads, 0, stream>>>(util::blasToCublas(uplo), um, un,
                                                         util::cppToCudaCast(a), to_uint(lda),
                                                         util::cppToCudaCast(b), to_uint(ldb));
  }
}

DLAF_CUBLAS_LACPY_ETI(, float);
DLAF_CUBLAS_LACPY_ETI(, double);
DLAF_CUBLAS_LACPY_ETI(, std::complex<float>);
DLAF_CUBLAS_LACPY_ETI(, std::complex<double>);
}
