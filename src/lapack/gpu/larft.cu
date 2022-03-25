//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cublas_v2.h>
#include "dlaf/common/assert.h"
#include "dlaf/cublas/api.h"
#include "dlaf/cublas/error.h"
#include "dlaf/cuda/assert.cu.h"
#include "dlaf/cuda/error.h"
#include "dlaf/lapack/gpu/larft.h"
#include "dlaf/types.h"
#include "dlaf/util_cublas.h"
#include "dlaf/util_cuda.h"
#include "dlaf/util_math.h"
#include "dlaf/util_matrix.h"

namespace dlaf::gpulapack {
namespace kernels {

using namespace dlaf::util::cuda_operators;

struct LarftGemvParams {
  static constexpr unsigned kernel_tile_size_rows_v = 64;
  static constexpr unsigned kernel_tile_size_rows_t = 32;
  static constexpr unsigned kernel_tile_size_cols_t = 32;
};

template <class T>
__global__ void larftGemv(const unsigned n, const unsigned k, const T* v, unsigned ldv, T* t,
                          unsigned ldt) {
  constexpr unsigned kts_rows_v = LarftGemvParams::kernel_tile_size_rows_v;
  constexpr unsigned kts_rows_t = LarftGemvParams::kernel_tile_size_rows_t;
  constexpr unsigned kts_cols_t = LarftGemvParams::kernel_tile_size_cols_t;
  // Note: kts_cols_v = kts_rows_t

  DLAF_GPU_ASSERT_HEAVY(kts_rows_t == blockDim.x);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.z);
  DLAF_GPU_ASSERT_HEAVY(ceilDiv(k, kts_rows_t) == gridDim.x);
  DLAF_GPU_ASSERT_HEAVY(ceilDiv(k, kts_cols_t) == gridDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == gridDim.z);

  constexpr unsigned lds = kts_rows_v + 1;
  __shared__ T shared[kts_rows_t * lds];

  const unsigned i_t = blockIdx.x * kts_rows_t;
  const unsigned j_t = blockIdx.y * kts_cols_t;

  // quick return if block covers only the lower part of t.
  if (i_t >= j_t + kts_cols_t - 1)
    return;

  // skip multiplications of 0s starting from j_t.
  for (unsigned l_v = j_t; l_v < n; l_v += kts_rows_v) {
    // copy the left part of V which partecipate to shared memory.
    const unsigned j_max = std::min(kts_rows_t, k - i_t);
    const unsigned l_max = std::min(kts_rows_v, n - l_v);

    for (unsigned j = 0; j < j_max; ++j) {
      for (unsigned l = threadIdx.x; l < l_max; l += kts_rows_t) {
        shared[l + j * lds] = conj(v[l_v + l + (j + i_t) * ldv]);
      }
    }
    __syncthreads();

    const unsigned i = i_t + threadIdx.x;
    const unsigned j_t_max = std::min(k, j_t + kts_cols_t);
    for (unsigned j = j_t; j < j_t_max; ++j) {
      if (i < j) {
        T& t_ij = t[i + j * ldt];
        T tau = t[j + j * ldt];
        for (unsigned l = 0; l < l_max; ++l) {
          if (l_v + l >= j) {
            t_ij = t_ij - tau * shared[l + threadIdx.x * lds] * v[l_v + l + j * ldv];
          }
        }
      }
    }
  }
}

struct LarftTrmvParams {
  static constexpr unsigned kernel_tile_size_rows_t = 512;
};

template <class T>
__global__ void larftTrmv(const unsigned k, T* t, unsigned ldt) {
  constexpr unsigned kts_rows_t = LarftTrmvParams::kernel_tile_size_rows_t;

  DLAF_GPU_ASSERT_HEAVY(kts_rows_t == blockDim.x);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.z);
  DLAF_GPU_ASSERT_HEAVY(1 == gridDim.x);
  DLAF_GPU_ASSERT_HEAVY(1 == gridDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == gridDim.z);

  for (unsigned ii = 0; ii < k; ii += kts_rows_t) {
    unsigned i = ii + threadIdx.x;
    if (i < k) {
      for (unsigned j = ii + 1; j < k; ++j) {
        T tmp{0};
        for (unsigned l = 0; l < j; ++l) {
          if (i <= l)
            tmp = tmp + t[i + l * ldt] * t[l + j * ldt];
        }
        __syncthreads();
        if (i < j)
          t[i + j * ldt] = tmp;
      }
    }
  }
}
}

template <class T>
void larft0(cublasHandle_t handle, const SizeType n, SizeType k, const T* v, const SizeType ldv,
            const T* tau, T* t, const SizeType ldt) {
  DLAF_ASSERT(n >= 0, n);
  DLAF_ASSERT(k >= 0, k);

  if (k == 0)
    return;

  cudaStream_t stream;
  DLAF_CUBLAS_CALL(cublasGetStream(handle, &stream));

  DLAF_CUDA_CALL(
      cudaMemset2DAsync(t, sizeof(T) * to_sizet(ldt), 0, sizeof(T) * to_sizet(k), to_sizet(k), stream));

  if (n == 0)
    return;

  // Reduce k excluding reflectors with size 0.
  k = std::min(n, k);

  // Note:
  // T factor is an upper triangular square matrix, built column by column
  // with taus values on the diagonal
  //
  // T(j,j) = tau(j)
  //
  // and in the upper triangular part the following formula applies
  //
  // T(0:j, j) = T(0:j, 0:j) . -tau(j) . V(j:, 0:j)* . V(j:, j)
  //
  //
  // The result is achieved in two main steps:
  // 1) t = -tau(j) . V(j:, 0:j)* . V(j:, j)
  // 2) T(0:j, j) = T(0:j, 0:j) . t

  // 1st step: compute the column partial result `t`
  // First we compute the matrix vector multiplication for each column
  // -tau(j) . V(j:, 0:j)* . V(j:, j)

  DLAF_CUDA_CALL(cudaMemcpy2DAsync(t, to_sizet(ldt + 1) * sizeof(T), tau, sizeof(T), sizeof(T),
                                   to_sizet(k), cudaMemcpyDefault, stream));

  const int n_ = to_int(n);
  const int k_ = to_int(k);
  const int ldv_ = to_int(ldv);
  const int ldt_ = to_int(ldt);

  auto v_ = [v, ldv_](int i, int j) { return util::blasToCublasCast(v + i + j * ldv_); };
  auto t_ = [t, ldt_](int i, int j) { return util::blasToCublasCast(t + i + j * ldt_); };

  for (int j = 0; j < k_; ++j) {
    const auto mtau = util::blasToCublasCast(-tau[j]);
    const auto one = util::blasToCublasCast(T{1});
    gpublas::Gemv<T>::call(handle, CUBLAS_OP_C, n_ - j, j, &mtau, v_(j, 0), ldv_, v_(j, j), 1, &one,
                           t_(0, j), 1);
  }

  // 2nd step: compute the T factor, by performing the last step on each column
  for (int j = 1; j < k_; ++j) {
    gpublas::Trmv<T>::call(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, j,
                           t_(0, 0), ldt_, t_(0, j), 1);
  }
}

template <class T>
void larft(cublasHandle_t handle, const SizeType n, SizeType k, const T* v, const SizeType ldv,
           const T* tau, T* t, const SizeType ldt) {
  DLAF_ASSERT(n >= 0, n);
  DLAF_ASSERT(k >= 0, k);

  if (k == 0)
    return;

  cudaStream_t stream;
  DLAF_CUBLAS_CALL(cublasGetStream(handle, &stream));

  DLAF_CUDA_CALL(
      cudaMemset2DAsync(t, sizeof(T) * to_sizet(ldt), 0, sizeof(T) * to_sizet(k), to_sizet(k), stream));

  if (n == 0)
    return;

  // Reduce k excluding reflectors with size 0.
  k = std::min(n, k);

  // Note:
  // T factor is an upper triangular square matrix, built column by column
  // with taus values on the diagonal
  //
  // T(j,j) = tau(j)
  //
  // and in the upper triangular part the following formula applies
  //
  // T(0:j, j) = T(0:j, 0:j) . -tau(j) . V(j:, 0:j)* . V(j:, j)
  //
  //
  // The result is achieved in two main steps:
  // 1) t = -tau(j) . V(j:, 0:j)* . V(j:, j)
  // 2) T(0:j, j) = T(0:j, 0:j) . t

  // 1st step: compute the column partial result `t`
  // First we compute the matrix vector multiplication for each column
  // -tau(j) . V(j:, 0:j)* . V(j:, j)

  DLAF_CUDA_CALL(cudaMemcpy2DAsync(t, to_sizet(ldt + 1) * sizeof(T), tau, sizeof(T), sizeof(T),
                                   to_sizet(k), cudaMemcpyDefault, stream));

  const unsigned un = to_uint(n);
  const unsigned uk = to_uint(k);
  const unsigned uldv = to_uint(ldv);
  const unsigned uldt = to_uint(ldt);

  {
    constexpr unsigned kts_rows_t = kernels::LarftGemvParams::kernel_tile_size_rows_t;
    constexpr unsigned kts_cols_t = kernels::LarftGemvParams::kernel_tile_size_cols_t;

    dim3 nr_threads(kts_rows_t);
    dim3 nr_blocks(util::ceilDiv(uk, kts_rows_t), util::ceilDiv(uk, kts_cols_t));
    kernels::larftGemv<<<nr_blocks, nr_threads, 0, stream>>>(un, uk, util::blasToCublasCast(v), uldv,
                                                             util::blasToCublasCast(t), uldt);
  }
  {
    // 2nd step: compute the T factor, by performing the last step on each column
    constexpr unsigned kts_rows_t = kernels::LarftTrmvParams::kernel_tile_size_rows_t;
    kernels::larftTrmv<<<1, kts_rows_t, 0, stream>>>(uk, util::blasToCublasCast(t), uldt);
  }
}

DLAF_CUBLAS_LARFT_ETI(, float);
DLAF_CUBLAS_LARFT_ETI(, double);
DLAF_CUBLAS_LARFT_ETI(, std::complex<float>);
DLAF_CUBLAS_LARFT_ETI(, std::complex<double>);
}
