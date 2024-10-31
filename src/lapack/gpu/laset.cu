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

#include <dlaf/gpu/assert.cu.h>
#include <dlaf/lapack/gpu/laset.h>
#include <dlaf/util_cublas.h>
#include <dlaf/util_cuda.h>
#include <dlaf/util_math.h>

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

constexpr std::size_t DIM = 32;

template <class T>
__global__ void gemv_conj_kernel_v1(int m, int n, const T alpha, const T* A, int lda, const T* x,
                                    const T beta, T* y) {
  const std::size_t j = threadIdx.y + blockIdx.y * DIM;

  if (j >= n)
    return;

  const T* a_col = &A[j * static_cast<std::size_t>(lda)];

  T resY;
  memset(&resY, 0, sizeof(resY));
  for (std::size_t i = 0; i < static_cast<std::size_t>(m); i++)
    resY = fma(conj(a_col[i]), x[i], resY);

  y[j] = fma(alpha, resY, beta * y[j]);
}

template <class T>
__global__ void gemv_conj_kernel_v2(int m, int n, const T alpha, const T* A, int lda, const T* x,
                                    const T beta, T* y) {
  const std::size_t j0 = threadIdx.x * 4 + blockIdx.x * blockDim.x;
  const std::size_t j1 = j0 + 1;
  const std::size_t j2 = j0 + 2;
  const std::size_t j3 = j0 + 3;

  T resY[4], xi;
  memset(resY, 0, sizeof(resY));
  for (std::size_t i = 0; i < static_cast<std::size_t>(m); i++) {
    xi = x[i];
    resY[0] = conj(A[i + j0 * static_cast<std::size_t>(lda)]) * xi + resY[0];
    resY[1] = conj(A[i + j1 * static_cast<std::size_t>(lda)]) * xi + resY[1];
    resY[2] = conj(A[i + j2 * static_cast<std::size_t>(lda)]) * xi + resY[2];
    resY[3] = conj(A[i + j3 * static_cast<std::size_t>(lda)]) * xi + resY[3];
  }
  y[j0] = alpha * resY[0] + beta * y[j0];
  y[j1] = alpha * resY[1] + beta * y[j1];
  y[j2] = alpha * resY[2] + beta * y[j2];
  y[j3] = alpha * resY[3] + beta * y[j3];
}

}

template <class T>
void laset(blas::Uplo uplo, SizeType m, SizeType n, T alpha, T beta, T* a, SizeType lda,
           whip::stream_t stream) {
  constexpr unsigned kernel_tile_size_rows = kernels::LasetParams::kernel_tile_size_rows;
  constexpr unsigned kernel_tile_size_cols = kernels::LasetParams::kernel_tile_size_cols;
  const unsigned um = to_uint(m);
  const unsigned un = to_uint(n);

  dim3 nr_threads(kernel_tile_size_rows, 1);
  dim3 nr_blocks(util::ceilDiv(um, kernel_tile_size_rows), util::ceilDiv(un, kernel_tile_size_cols));
  kernels::laset<<<nr_blocks, nr_threads, 0, stream>>>(util::blasToCublas(uplo), um, un,
                                                       util::cppToCudaCast(alpha),
                                                       util::cppToCudaCast(beta), util::cppToCudaCast(a),
                                                       to_uint(lda));
}

DLAF_CUBLAS_LASET_ETI(, float);
DLAF_CUBLAS_LASET_ETI(, double);
DLAF_CUBLAS_LASET_ETI(, std::complex<float>);
DLAF_CUBLAS_LASET_ETI(, std::complex<double>);

template <class T>
void gemv_conj_gpu(int m, int n, const T alpha, const T* A, int lda, const T* x, const T beta, T* y,
                   whip::stream_t stream) {
  // V1
  dim3 blocks(1, util::ceilDiv(to_sizet(n), kernels::DIM));
  dim3 threads(1, kernels::DIM);

  // V2
  // dim3 blocks(util::ceilDiv(to_sizet(m), 4 * kernels::DIM), 1);
  // dim3 threads(kernels::DIM, 1);

  kernels::gemv_conj_kernel_v1<<<blocks, threads, 0, stream>>>(
      m, n, util::cppToCudaCast(alpha), util::cppToCudaCast(A), lda, util::cppToCudaCast(x),
      util::cppToCudaCast(beta), util::cppToCudaCast(y));
}

DLAF_CUSTOM_GEMV_ETI(, float);
DLAF_CUSTOM_GEMV_ETI(, double);
DLAF_CUSTOM_GEMV_ETI(, std::complex<float>);
DLAF_CUSTOM_GEMV_ETI(, std::complex<double>);

}
