//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cublas_v2.h>
#include <whip.hpp>

#include <dlaf/blas/tile.h>
#include <dlaf/common/assert.h>
#include <dlaf/gpu/assert.cu.h>
#include <dlaf/lapack/gpu/larft.h>
#include <dlaf/types.h>
#include <dlaf/util_cublas.h>
#include <dlaf/util_math.h>

namespace dlaf::gpulapack {
namespace kernels {

using namespace dlaf::util::cuda_operators;

struct LarftGemvParams {
  static constexpr unsigned kernel_tile_size_rows_v = 64;
  static constexpr unsigned kernel_tile_size_rows_t = 32;
  static constexpr unsigned kernel_tile_size_cols_t = 32;
};

template <class T>
__global__ void larft_gemv(const unsigned n, const unsigned k, const T* v, unsigned ldv, T* t,
                           unsigned ldt) {
  constexpr unsigned kts_rows_t = LarftGemvParams::kernel_tile_size_rows_t;
  constexpr unsigned kts_cols_t = LarftGemvParams::kernel_tile_size_cols_t;

  // Note: kts_cols_v = kts_rows_t
  constexpr unsigned kts_rows_v = LarftGemvParams::kernel_tile_size_rows_v;
  constexpr unsigned kts_cols_v = kts_rows_t;

  DLAF_GPU_ASSERT_HEAVY(kts_rows_t == blockDim.x);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.z);
  DLAF_GPU_ASSERT_HEAVY(ceilDiv(k, kts_rows_t) == gridDim.x);
  DLAF_GPU_ASSERT_HEAVY(ceilDiv(k, kts_cols_t) == gridDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == gridDim.z);

  constexpr unsigned lds = kts_rows_v + 1;
  __shared__ T shared[kts_rows_t * lds];

  const unsigned i_t_b = blockIdx.x * kts_rows_t;
  const unsigned j_t_b = blockIdx.y * kts_cols_t;

  // quick return if block covers only the lower part of t.
  if (i_t_b >= j_t_b + kts_cols_t - 1)
    return;

  // Note: each iteration computes a chunk of V that fits in shared memory
  for (unsigned l_v = 0; l_v < n; l_v += kts_rows_v) {
    const unsigned i_va = l_v;
    const unsigned j_va = i_t_b;

    const unsigned va_rows = std::min(kts_rows_v, n - i_va);
    const unsigned va_cols = std::min(kts_cols_v, k - j_va);  // TODO it can be computed outside loop

    // load current chunk of row of tiles into shared memory
    // each thread loads a set of rows and stores it conj (not yet transposed) into shared memory
    for (unsigned j = 0; j < va_cols; ++j) {
      for (unsigned i = threadIdx.x; i < va_rows; ++i) {
        shared[i + j * lds] = conj(v[i_va + i + ldv * (j_va + j)]);
      }
    }

    __syncthreads();

    const unsigned i_t = i_t_b + threadIdx.x;
    const unsigned j_t_end = std::min(k, j_t_b + kts_cols_t);
    for (unsigned j_t = j_t_b; j_t < j_t_end; ++j_t) {
      // Note: it might be upper or general
      if (i_t < j_t) {
        const T tau = t[j_t + j_t * ldt];
        T t_ij = t[i_t + j_t * ldt];
        // loop over the entire chunk of the block
        for (unsigned l = 0; l < va_rows; ++l) {
          t_ij = t_ij - tau * shared[l + threadIdx.x * lds] * v[l_v + l + j_t * ldv];
        }
        t[i_t + j_t * ldt] = t_ij;
      }
    }
  }
}
}

template <class T>
void larft_gemv0(cublasHandle_t handle, const SizeType m, const SizeType k, const T* v, const SizeType ldv, const T* tau, T* t, const SizeType ldt) {
  DLAF_ASSERT(m >= 0, m);
  DLAF_ASSERT(k >= 0, k);
  DLAF_ASSERT(ldv >= m, ldv, m);
  DLAF_ASSERT(ldt >= k, ldt, k);

  const int m_ = to_int(m);
  const int k_ = to_int(k);
  const int ldv_ = to_int(ldv);
  const int ldt_ = to_int(ldt);

  auto v_ = [v, ldv_](int i, int j) { return util::blasToCublasCast(v + i + j * ldv_); };
  auto t_ = [t, ldt_](int i, int j) { return util::blasToCublasCast(t + i + j * ldt_); };

  for (int j = 1; j < k_; ++j) {
    const auto mtau = util::blasToCublasCast(-tau[j]);
    const auto one = util::blasToCublasCast(T{1});
    gpublas::internal::Gemv<T>::call(handle, CUBLAS_OP_C, m, j, &mtau, v_(0, 0), ldv_, v_(0, j), 1, &one,
                           t_(0, j), 1);
  }
}

template <class T>
void larft_gemv1(const SizeType m, const SizeType k, const T* v, const SizeType ldv, T* t,
                const SizeType ldt, whip::stream_t stream) {
  DLAF_ASSERT(m >= 0, m);
  DLAF_ASSERT(k >= 0, k);
  DLAF_ASSERT(ldv >= m, ldv, m);
  DLAF_ASSERT(ldt >= k, ldt, k);

  if (m == 0)
    return;

  if (k == 0)
    return;

  const unsigned um = to_uint(m);
  const unsigned uk = to_uint(k);
  const unsigned uldv = to_uint(ldv);
  const unsigned uldt = to_uint(ldt);

  constexpr unsigned kts_rows_t = kernels::LarftGemvParams::kernel_tile_size_rows_t;
  constexpr unsigned kts_cols_t = kernels::LarftGemvParams::kernel_tile_size_cols_t;

  dim3 nr_threads(kts_rows_t);
  dim3 nr_blocks(util::ceilDiv(uk, kts_rows_t), util::ceilDiv(uk, kts_cols_t));

  kernels::larft_gemv<<<nr_blocks, nr_threads, 0, stream>>>(um, uk, util::blasToCublasCast(v),
                                                                  uldv, util::blasToCublasCast(t), uldt);
}

DLAF_CUBLAS_LARFT_GEMV_ETI(, float);
DLAF_CUBLAS_LARFT_GEMV_ETI(, double);
DLAF_CUBLAS_LARFT_GEMV_ETI(, std::complex<float>);
DLAF_CUBLAS_LARFT_GEMV_ETI(, std::complex<double>);

}
