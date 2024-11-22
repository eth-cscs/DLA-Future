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
#include <dlaf/lapack/gpu/add.h>
#include <dlaf/lapack/gpu/lacpy.h>
#include <dlaf/lapack/gpu/larft.h>
#include <dlaf/lapack/gpu/laset.h>
#include <dlaf/types.h>
#include <dlaf/util_cublas.h>
#include <dlaf/util_math.h>

namespace dlaf::gpulapack {
namespace kernels {

using namespace dlaf::util::cuda_operators;

template <unsigned kts_rows_t, unsigned kts_cols_t, class T>
    __global__ void fix_tau(const unsigned k, const T* tau, unsigned inctau, T* t, unsigned ldt) {
  DLAF_GPU_ASSERT_HEAVY(kts_rows_t == blockDim.x);
  DLAF_GPU_ASSERT_HEAVY(kts_cols_t == blockDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.z);
  DLAF_GPU_ASSERT_HEAVY(ceilDiv(k, kts_rows_t) == gridDim.x);
  DLAF_GPU_ASSERT_HEAVY(ceilDiv(k, kts_cols_t) == gridDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == gridDim.z);

  const unsigned i = blockIdx.x * kts_rows_t + threadIdx.x;
  const unsigned j = blockIdx.y * kts_cols_t + threadIdx.y;

  // quick return if outside of t.
  if (i >= k || j >= k)
    return;

  T& t_ij = t[i + j * ldt];
  T tau_j = tau[j * inctau];
  if (i > j)
    t_ij = {0.};
  else if (i == j)
    t_ij = tau_j;
  else
    t_ij = - tau_j * t_ij;
}

template <unsigned kts_k, unsigned kts_rows_t, unsigned kts_cols_t, class T>
__global__ void larft_gemv10(const unsigned m, const unsigned k, const T* v, unsigned ldv, T* t,
                             unsigned ldt) {
  DLAF_GPU_ASSERT_HEAVY(kts_rows_t == blockDim.x);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.z);
  DLAF_GPU_ASSERT_HEAVY(ceilDiv(k, kts_rows_t) == gridDim.x);
  DLAF_GPU_ASSERT_HEAVY(ceilDiv(k, kts_cols_t) == gridDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == gridDim.z);

  constexpr unsigned lds = kts_k + 1;
  __shared__ T shared[kts_rows_t * lds];

  const unsigned i_b = blockIdx.x * kts_rows_t;
  const unsigned j_b = blockIdx.y * kts_cols_t;
  const unsigned j_end = std::min(k, j_b + kts_cols_t);

  const unsigned i_t = threadIdx.x;
  const unsigned i = i_b + i_t;
  // quick return if block covers only the lower part of t.
  if (i_b >= j_b + kts_cols_t - 1)
    return;

  // Note: each iteration computes a chunk of V that fits in shared memory
  for (unsigned l_b = 0; l_b < m; l_b += kts_k) {
    const unsigned l_t_end = std::min(kts_k, m - l_b);
    const unsigned i_t_end = std::min(kts_rows_t, k - i_b);

    // load conj(v(l_b:l_end,i_b:i_end)) into shared memory
    // where l_end = l_b + l_t_end, i_end = i_b + i_t_end.
    for (unsigned ii_t = 0; ii_t < i_t_end; ++ii_t) {
      for (unsigned l_t = threadIdx.x; l_t < l_t_end; l_t += blockDim.x) {
        const unsigned l = l_b + l_t;
        const unsigned i = i_b + ii_t;
        shared[l_t + ii_t * lds] = conj(v[l + i * ldv]);
      }
    }

    __syncthreads();

    for (unsigned j = j_b; j < j_end; ++j) {
      // Note: it might be upper or general
      if (i < j) {
        const T tau = t[j + j * ldt];
        T t_ij = t[i + j * ldt];
        // loop over the entire chunk of the block
        for (unsigned l_t = 0; l_t < l_t_end; ++l_t) {
          const unsigned l = l_b + l_t;
          t_ij = t_ij - tau * shared[l_t + i_t * lds] * v[l + j * ldv];
        }
        t[i + j * ldt] = t_ij;
      }
    }

    __syncthreads();
  }
}

template <unsigned kts_k, unsigned kts_rows_t, unsigned kts_cols_t, class T>
__global__ void larft_gemv11(const unsigned m, const unsigned k, const T* v, unsigned ldv, T* t,
                             unsigned ldt) {
  DLAF_GPU_ASSERT_HEAVY(kts_rows_t == blockDim.x);
  DLAF_GPU_ASSERT_HEAVY(kts_cols_t == blockDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.z);
  DLAF_GPU_ASSERT_HEAVY(ceilDiv(k, kts_rows_t) == gridDim.x);
  DLAF_GPU_ASSERT_HEAVY(ceilDiv(k, kts_cols_t) == gridDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == gridDim.z);

  constexpr unsigned lds = kts_k + 1;
  __shared__ T shared[kts_rows_t * lds];

  const unsigned i_b = blockIdx.x * kts_rows_t;
  const unsigned j_b = blockIdx.y * kts_cols_t;

  const unsigned i_t = threadIdx.x;
  const unsigned j_t = threadIdx.y;

  const unsigned i = i_b + i_t;
  const unsigned j = j_b + j_t;

  // quick return if block covers only the lower part of t.
  if (i_b >= j_b + kts_cols_t - 1)
    return;

  T tmp_ij = {0.};

  // Note: each iteration computes a chunk of V that fits in shared memory
  for (unsigned l_b = 0; l_b < m; l_b += kts_k) {
    const unsigned l_t_end = std::min(kts_k, m - l_b);
    const unsigned i_t_end = std::min(kts_rows_t, k - i_b);

    // load conj(v(l_b:l_end,i_b:i_end)) into shared memory
    // where l_end = l_b + l_t_end, i_end = i_b + i_t_end.
    for (unsigned ii_t = threadIdx.y; ii_t < i_t_end; ii_t += blockDim.y) {
      for (unsigned l_t = threadIdx.x; l_t < l_t_end; l_t += blockDim.x) {
        const unsigned l = l_b + l_t;
        const unsigned i = i_b + ii_t;
        shared[l_t + ii_t * lds] = conj(v[l + i * ldv]);
      }
    }

    __syncthreads();

    if (i < j && j < k) {
      // loop over the entire chunk of the block
      for (unsigned l_t = 0; l_t < l_t_end; ++l_t) {
        const unsigned l = l_b + l_t;
        tmp_ij = fma(shared[l_t + i_t * lds], v[l + j * ldv], tmp_ij);
      }
    }

    __syncthreads();
  }
  if (i < j && j < k) {
    const T tau = t[j + j * ldt];
    T t_ij = t[i + j * ldt];
    t[i + j * ldt] = fma(-tau, tmp_ij, t_ij);
  }
}

template <unsigned kts_k, unsigned kts_rows_t, unsigned kts_cols_t, class T>
__global__ void larft_gemv12(const unsigned m, const unsigned k, const T* v, unsigned ldv, T* t,
                             unsigned ldt) {
  DLAF_GPU_ASSERT_HEAVY(kts_rows_t == blockDim.x);
  DLAF_GPU_ASSERT_HEAVY(kts_cols_t == blockDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.z);
  DLAF_GPU_ASSERT_HEAVY(ceilDiv(k, kts_rows_t) == gridDim.x);
  DLAF_GPU_ASSERT_HEAVY(ceilDiv(k, kts_cols_t) == gridDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == gridDim.z);

  constexpr unsigned lds = kts_k + 1;
  __shared__ T va[kts_rows_t * lds];
  __shared__ T vb[kts_cols_t * lds];

  const unsigned i_b = blockIdx.x * kts_rows_t;
  const unsigned j_b = blockIdx.y * kts_cols_t;
 
  const unsigned i_t_end = std::min(kts_rows_t, k - i_b);
  const unsigned j_t_end = std::min(kts_cols_t, k - j_b);

  const unsigned i_t = threadIdx.x;
  const unsigned j_t = threadIdx.y;

  const unsigned i = i_b + i_t;
  const unsigned j = j_b + j_t;

  // quick return if block covers only the lower part of t.
  if (i_b >= j_b + kts_cols_t - 1)
    return;

  T tmp_ij = {0.};

  // Note: each iteration computes a chunk of V that fits in shared memory
  for (unsigned l_b = 0; l_b < m; l_b += kts_k) {
    const unsigned l_t_end = std::min(kts_k, m - l_b);

    // load conj(v(l_b:l_end,i_b:i_end)) into shared memory
    // where l_end = l_b + l_t_end, i_end = i_b + i_t_end.
    for (unsigned ii_t = threadIdx.y; ii_t < i_t_end; ii_t += blockDim.y) {
      for (unsigned l_t = threadIdx.x; l_t < l_t_end; l_t += blockDim.x) {
        const unsigned l = l_b + l_t;
        const unsigned i = i_b + ii_t;
        va[l_t + ii_t * lds] = v[l + i * ldv];
      }
    }

    // load conj(v(l_b:l_end,j_b:j_end)) into shared memory
    // where l_end = l_b + l_t_end, j_end = j_b + j_t_end.
    for (unsigned jj_t = threadIdx.y; jj_t < j_t_end; jj_t += blockDim.y) {
      for (unsigned l_t = threadIdx.x; l_t < l_t_end; l_t += blockDim.x) {
        const unsigned l = l_b + l_t;
        const unsigned j = j_b + jj_t;
        vb[l_t + jj_t * lds] = v[l + j * ldv];
      }
    }

    __syncthreads();

    T* v_lj = vb + j_t * lds;
    T* v_li = va + i_t * lds;

    // loop over the entire chunk of the block
    // Note: ij elements outside boundaries are computed and discarded 
    //       No access outside allocated shared memory.
    for (unsigned l_t = 0; l_t < l_t_end; ++l_t) {
      T cv_li = conj(*v_li);
      tmp_ij = fma(cv_li, *v_lj, tmp_ij);
      ++v_li;
      ++v_lj;
    }

    __syncthreads();
  }

  if (i < j && j < k) {
    const T tau = t[j + j * ldt];
    T& t_ij = t[i + j * ldt];
    t_ij = fma(-tau, tmp_ij, t_ij);
  }
}

template <unsigned kts_k, unsigned kts_rows_t, unsigned kts_cols_t, class T>
__global__ void larft_gemv20(const unsigned m, const unsigned k, const T* v, unsigned ldv, T* t,
                             unsigned ldt) {
  DLAF_GPU_ASSERT_HEAVY(kts_rows_t == blockDim.x);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.z);
  DLAF_GPU_ASSERT_HEAVY(ceilDiv(k, kts_rows_t) == gridDim.x);
  DLAF_GPU_ASSERT_HEAVY(ceilDiv(k, kts_cols_t) == gridDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == gridDim.z);

  constexpr unsigned lds = kts_k + 1;
  __shared__ T shared[kts_rows_t * lds];

  const unsigned i_b = blockIdx.x * kts_rows_t;
  const unsigned j_b = blockIdx.y * kts_cols_t;

  const unsigned i_t_end = std::min(kts_rows_t, k - i_b);
  const unsigned j_t_end = std::min(kts_cols_t, k - j_b);

  const unsigned i_t = threadIdx.x;
  const unsigned i = i_b + i_t;

  // quick return if block covers only the lower part of t.
  if (i_b >= j_b + kts_cols_t - 1)
    return;

  T tmp_i[kts_cols_t] = {0.};

  // Note: each iteration computes a chunk of V that fits in shared memory
  for (unsigned l_b = 0; l_b < m; l_b += kts_k) {
    const unsigned l_t_end = std::min(kts_k, m - l_b);

    // load conj(v(l_b:l_end,i_b:i_end)) into shared memory
    // where l_end = l_b + l_t_end, i_end = i_b + i_t_end.
    for (unsigned ii_t = 0; ii_t < i_t_end; ++ii_t) {
      for (unsigned l_t = threadIdx.x; l_t < l_t_end; l_t += blockDim.x) {
        const unsigned l = l_b + l_t;
        const unsigned i = i_b + ii_t;
        shared[l_t + ii_t * lds] = v[l + i * ldv];
      }
    }

    __syncthreads();

    // Note: The values on and below the diagonal are computed anyway but discarded in the next step.
    //       They access allocated shared memory but not initialized.
    for (unsigned l_t = 0; l_t < l_t_end; ++l_t) {
      const unsigned l = l_b + l_t;
      T v_li = conj(shared[l_t + i_t * lds]);

      for (unsigned j_t = 0; j_t < j_t_end; ++j_t) {
        const unsigned j = j_b + j_t;
        T v_lj = v[l + j * ldv];
        tmp_i[j_t] = fma(v_li, v_lj, tmp_i[j_t]);
      }
    }

    __syncthreads();
  }

  for (unsigned j_t = 0; j_t < j_t_end; ++j_t) {
    unsigned j = j_b + j_t;
    if (i < j) {
      T tau = t[j + j * ldt];
      T& t_ij = t[i + j * ldt];
      t_ij = fma(-tau, tmp_i[j_t], t_ij);
    }
  }
}

template <unsigned kts_k, unsigned kts_rows_t, unsigned kts_cols_t, unsigned cols, class T>
__global__ void larft_gemv21(const unsigned m, const unsigned k, const T* v, unsigned ldv, T* t,
                             unsigned ldt) {
  DLAF_GPU_ASSERT_HEAVY(kts_rows_t == blockDim.x);
  DLAF_GPU_ASSERT_HEAVY(kts_cols_t == blockDim.y * cols);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.z);
  DLAF_GPU_ASSERT_HEAVY(ceilDiv(k, kts_rows_t) == gridDim.x);
  DLAF_GPU_ASSERT_HEAVY(ceilDiv(k, kts_cols_t) == gridDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == gridDim.z);

  constexpr unsigned lds = kts_k + 1;
  __shared__ T shared[kts_rows_t * lds];

  const unsigned i_b = blockIdx.x * kts_rows_t;
  const unsigned j_b = blockIdx.y * kts_cols_t;
  const unsigned j_bt = j_b + threadIdx.y * cols;

  const unsigned i_t_end = std::min(kts_rows_t, k - i_b);
  const unsigned j_tt_end = std::min(cols, k - j_bt);

  const unsigned i_t = threadIdx.x;
  const unsigned i = i_b + i_t;

  // quick return if block covers only the lower part of t.
  if (i_b >= j_b + kts_cols_t - 1)
    return;

  T tmp_i[cols] = {0.};

  // Note: each iteration computes a chunk of V that fits in shared memory
  for (unsigned l_b = 0; l_b < m; l_b += kts_k) {
    const unsigned l_t_end = std::min(kts_k, m - l_b);
    const unsigned i_t_end = std::min(kts_rows_t, k - i_b);

    // load conj(v(l_b:l_end,i_b:i_end)) into shared memory
    // where l_end = l_b + l_t_end, i_end = i_b + i_t_end.
    for (unsigned ii_t = threadIdx.y; ii_t < i_t_end; ii_t += blockDim.y) {
      for (unsigned l_t = threadIdx.x; l_t < l_t_end; l_t += blockDim.x) {
        const unsigned l = l_b + l_t;
        const unsigned i = i_b + ii_t;
        shared[l_t + ii_t * lds] = v[l + i * ldv];
      }
    }

    __syncthreads();

    // Note: The values on and below the diagonal are computed anyway but discarded in the next step.
    //       They access allocated shared memory but not initialized.
    for (unsigned l_t = 0; l_t < l_t_end; ++l_t) {
      const unsigned l = l_b + l_t;
      T v_li = conj(shared[l_t + i_t * lds]);

      for (unsigned j_tt = 0; j_tt < j_tt_end; ++j_tt) {
        const unsigned j = j_bt + j_tt;
        T v_lj = v[l + j * ldv];
        tmp_i[j_tt] = fma(v_li, v_lj, tmp_i[j_tt]);
      }
    }

    __syncthreads();
  }

  for (unsigned j_tt = 0; j_tt < j_tt_end; ++j_tt) {
    unsigned j = j_bt + j_tt;
    if (i < j) {
      T tau = t[j + j * ldt];
      T& t_ij = t[i + j * ldt];
      t_ij = fma(-tau, tmp_i[j_tt], t_ij);
    }
  }
}

template <unsigned kts_k, unsigned kts_rows_t, unsigned kts_cols_t, unsigned cols, class T>
__global__ void larft_gemv22(const unsigned m, const unsigned k, const T* v, unsigned ldv, T* t,
                             unsigned ldt) {
  DLAF_GPU_ASSERT_HEAVY(kts_rows_t == blockDim.x);
  DLAF_GPU_ASSERT_HEAVY(kts_cols_t == blockDim.y * cols);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.z);
  DLAF_GPU_ASSERT_HEAVY(ceilDiv(k, kts_rows_t) == gridDim.x);
  DLAF_GPU_ASSERT_HEAVY(ceilDiv(k, kts_cols_t) == gridDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == gridDim.z);

  constexpr unsigned lds = kts_k + 1;
  __shared__ T va[kts_rows_t * lds];
  __shared__ T vb[kts_cols_t * lds];

  const unsigned i_b = blockIdx.x * kts_rows_t;
  const unsigned j_b = blockIdx.y * kts_cols_t;
 
  const unsigned j_bt = j_b + threadIdx.y * cols;

  const unsigned i_t_end = std::min(kts_rows_t, k - i_b);
  const unsigned j_t_end = std::min(kts_cols_t, k - j_b);
  const unsigned j_tt_end = std::min(cols, k - j_bt);

  const unsigned i_t = threadIdx.x;
  const unsigned i = i_b + i_t;

  T* vb_t = vb + threadIdx.y * cols * lds;

  // quick return if block covers only the lower part of t.
  if (i_b >= j_b + kts_cols_t - 1)
    return;

  T tmp_i[cols] = {0.};

  // Note: each iteration computes a chunk of V that fits in shared memory
  for (unsigned l_b = 0; l_b < m; l_b += kts_k) {
    const unsigned l_t_end = std::min(kts_k, m - l_b);

    // load conj(v(l_b:l_end,i_b:i_end)) into shared memory
    // where l_end = l_b + l_t_end, i_end = i_b + i_t_end.
    for (unsigned ii_t = threadIdx.y; ii_t < i_t_end; ii_t += blockDim.y) {
      for (unsigned l_t = threadIdx.x; l_t < l_t_end; l_t += blockDim.x) {
        const unsigned l = l_b + l_t;
        const unsigned i = i_b + ii_t;
        va[l_t + ii_t * lds] = v[l + i * ldv];
      }
    }

    // load conj(v(l_b:l_end,j_b:j_end)) into shared memory
    // where l_end = l_b + l_t_end, j_end = j_b + j_t_end.
    for (unsigned j_t = threadIdx.y; j_t < j_t_end; j_t += blockDim.y) {
      for (unsigned l_t = threadIdx.x; l_t < l_t_end; l_t += blockDim.x) {
        const unsigned l = l_b + l_t;
        const unsigned j = j_b + j_t;
        vb[l_t + j_t * lds] = v[l + j * ldv];
      }
    }

    __syncthreads();

    // Note: The values on and below the diagonal are computed anyway but discarded in the next step.
    //       They access allocated shared memory but not initialized.
    T* v_i = va + i_t * lds;
    for (unsigned l_t = 0; l_t < l_t_end; ++l_t) {
      T v_li = conj(*v_i);
      ++v_i;
      T* v_lj = vb_t + l_t;
      T* tmp_ij = tmp_i;

      for (unsigned j_tt = 0; j_tt < j_tt_end; ++j_tt) {
        *tmp_ij = fma(v_li, *v_lj, *tmp_ij);
        v_lj += lds;
        ++tmp_ij;
      }
    }

    __syncthreads();
  }

  for (unsigned j_tt = 0; j_tt < j_tt_end; ++j_tt) {
    unsigned j = j_bt + j_tt;
    if (i < j) {
      T tau = t[j + j * ldt];
      T& t_ij = t[i + j * ldt];
      t_ij = fma(-tau, tmp_i[j_tt], t_ij);
    }
  }
}
}

template <class T>
void larft_gemv0(cublasHandle_t handle, const SizeType m, const SizeType k, const T* v,
                 const SizeType ldv, const T* tau, T* t, const SizeType ldt) {
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
void larft_gemv1_notau(cublasHandle_t handle, const SizeType m, const SizeType k, const T* v,
                        const SizeType ldv, T* t, const SizeType ldt) {
  DLAF_ASSERT(m >= 0, m);
  DLAF_ASSERT(k >= 0, k);
  DLAF_ASSERT(ldv >= m, ldv, m);
  DLAF_ASSERT(ldt >= k, ldt, k);

  const int m_ = to_int(m);
  const int k_ = to_int(k);
  const int ldv_ = to_int(ldv);
  const int ldt_ = to_int(ldt);

  auto v_ = util::blasToCublasCast(v);
  auto t_ = util::blasToCublasCast(t);

  const auto one = util::blasToCublasCast(T{1});
  gpublas::internal::Gemm<T>::call(handle, CUBLAS_OP_C, CUBLAS_OP_N, k, k, m, &one, v_, ldv_, v_, ldv_, &one, t_, ldt_);
}

template <class T>
void larft_gemv1_fixtau(const SizeType k, const T* tau, const SizeType inctau, T* t, const SizeType ldt, whip::stream_t stream) {
  constexpr unsigned kts_rows_t = 32;
  constexpr unsigned kts_cols_t = 32;

  DLAF_ASSERT(k >= 0, k);
  DLAF_ASSERT(ldt >= k, ldt, k);
  DLAF_ASSERT(inctau >= 1, inctau);

  if (k == 0)
    return;

  const unsigned uk = to_uint(k);
  const unsigned uinctau = to_uint(inctau);
  const unsigned uldt = to_uint(ldt);

  dim3 nr_threads(kts_rows_t, kts_cols_t);
  dim3 nr_blocks(util::ceilDiv(uk, kts_rows_t), util::ceilDiv(uk, kts_cols_t));

  auto tau_ = util::blasToCublasCast(tau);
  auto t_ = util::blasToCublasCast(t);
  kernels::fix_tau<kts_rows_t, kts_cols_t><<<nr_blocks, nr_threads, 0, stream>>>(uk, tau_, uinctau, t_, uldt);
}

namespace internal {

template <unsigned kts_k, unsigned kts_rows_t, unsigned kts_cols_t, class T>
void larft_gemv10(const SizeType m, const SizeType k, const T* v, const SizeType ldv, T* t,
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

  dim3 nr_threads(kts_rows_t);
  dim3 nr_blocks(util::ceilDiv(uk, kts_rows_t), util::ceilDiv(uk, kts_cols_t));

  kernels::larft_gemv10<kts_k, kts_rows_t, kts_cols_t>
      <<<nr_blocks, nr_threads, 0, stream>>>(um, uk, util::blasToCublasCast(v), uldv,
                                             util::blasToCublasCast(t), uldt);
}

template <unsigned kts_k, unsigned kts_rows_t, unsigned kts_cols_t, class T>
void larft_gemv11(const SizeType m, const SizeType k, const T* v, const SizeType ldv, T* t,
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

  dim3 nr_threads(kts_rows_t, kts_cols_t);
  dim3 nr_blocks(util::ceilDiv(uk, kts_rows_t), util::ceilDiv(uk, kts_cols_t));

  kernels::larft_gemv11<kts_k, kts_rows_t, kts_cols_t>
      <<<nr_blocks, nr_threads, 0, stream>>>(um, uk, util::blasToCublasCast(v), uldv,
                                             util::blasToCublasCast(t), uldt);
}

template <unsigned kts_k, unsigned kts_rows_t, unsigned kts_cols_t, class T>
void larft_gemv12(const SizeType m, const SizeType k, const T* v, const SizeType ldv, T* t,
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

  dim3 nr_threads(kts_rows_t, kts_cols_t);
  dim3 nr_blocks(util::ceilDiv(uk, kts_rows_t), util::ceilDiv(uk, kts_cols_t));

  kernels::larft_gemv12<kts_k, kts_rows_t, kts_cols_t>
      <<<nr_blocks, nr_threads, 0, stream>>>(um, uk, util::blasToCublasCast(v), uldv,
                                             util::blasToCublasCast(t), uldt);
}

template <unsigned kts_k, unsigned kts_rows_t, unsigned kts_cols_t, class T>
void larft_gemv20(const SizeType m, const SizeType k, const T* v, const SizeType ldv, T* t,
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

  dim3 nr_threads(kts_rows_t);
  dim3 nr_blocks(util::ceilDiv(uk, kts_rows_t), util::ceilDiv(uk, kts_cols_t));

  kernels::larft_gemv20<kts_k, kts_rows_t, kts_cols_t>
      <<<nr_blocks, nr_threads, 0, stream>>>(um, uk, util::blasToCublasCast(v), uldv,
                                             util::blasToCublasCast(t), uldt);
}

template <unsigned kts_k, unsigned kts_rows_t, unsigned kts_cols_t, unsigned cols, class T>
void larft_gemv21(const SizeType m, const SizeType k, const T* v, const SizeType ldv, T* t,
                  const SizeType ldt, whip::stream_t stream) {
  static_assert(kts_cols_t % cols == 0);

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

  dim3 nr_threads(kts_rows_t, kts_cols_t / cols);
  dim3 nr_blocks(util::ceilDiv(uk, kts_rows_t), util::ceilDiv(uk, kts_cols_t));

  kernels::larft_gemv21<kts_k, kts_rows_t, kts_cols_t, cols>
      <<<nr_blocks, nr_threads, 0, stream>>>(um, uk, util::blasToCublasCast(v), uldv,
                                             util::blasToCublasCast(t), uldt);
}

template <unsigned kts_k, unsigned kts_rows_t, unsigned kts_cols_t, unsigned cols, class T>
void larft_gemv22(const SizeType m, const SizeType k, const T* v, const SizeType ldv, T* t,
                  const SizeType ldt, whip::stream_t stream) {
  static_assert(kts_cols_t % cols == 0);

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

  dim3 nr_threads(kts_rows_t, kts_cols_t / cols);
  dim3 nr_blocks(util::ceilDiv(uk, kts_rows_t), util::ceilDiv(uk, kts_cols_t));

  kernels::larft_gemv22<kts_k, kts_rows_t, kts_cols_t, cols>
      <<<nr_blocks, nr_threads, 0, stream>>>(um, uk, util::blasToCublasCast(v), uldv,
                                             util::blasToCublasCast(t), uldt);
}
}


#define KERNEL_CALL_3(kid, variant, kts_k, kts_rows_t, kts_cols_t) \
  template <class T> \
  void larft_gemv##kid##variant(const SizeType m, const SizeType k, const T* v, const SizeType ldv, T* t, \
                                const SizeType ldt, whip::stream_t stream) { \
    internal::larft_gemv##kid<kts_k, kts_rows_t, kts_cols_t>(m, k, v, ldv, t, ldt, stream); \
  }

#define KERNEL_CALL_4(kid, variant, kts_k, kts_rows_t, kts_cols_t, cols) \
  template <class T> \
  void larft_gemv##kid##variant(const SizeType m, const SizeType k, const T* v, const SizeType ldv, T* t, \
                                const SizeType ldt, whip::stream_t stream) { \
    internal::larft_gemv##kid<kts_k, kts_rows_t, kts_cols_t, cols>(m, k, v, ldv, t, ldt, stream); \
  }

KERNEL_CALL_3(10, 00, 64, 32, 32)
KERNEL_CALL_3(10, 01, 64, 32, 8)
KERNEL_CALL_3(10, 02, 64, 32, 4)
KERNEL_CALL_3(10, 03, 64, 32, 1)

KERNEL_CALL_3(11, 00, 64, 32, 32)
KERNEL_CALL_3(11, 01, 64, 16, 16)

KERNEL_CALL_3(12, 00, 32, 32, 32)
KERNEL_CALL_3(12, 01, 32, 16, 16)
KERNEL_CALL_3(12, 02, 64, 16, 16)
KERNEL_CALL_3(12, 03, 64, 8, 8)
KERNEL_CALL_3(12, 04, 128, 8, 8)

KERNEL_CALL_3(20, 00, 64, 32, 32)
KERNEL_CALL_3(20, 01, 64, 32, 8)
KERNEL_CALL_3(20, 02, 64, 32, 4)
KERNEL_CALL_3(20, 03, 64, 32, 1)

KERNEL_CALL_4(21, 00, 64, 32, 32, 2)
KERNEL_CALL_4(21, 01, 64, 32, 32, 4)
KERNEL_CALL_4(21, 02, 64, 32, 32, 8)
KERNEL_CALL_4(21, 03, 64, 32, 32, 16)
KERNEL_CALL_4(21, 04, 64, 16, 32, 2)

KERNEL_CALL_4(22, 00, 32, 32, 32, 1)
KERNEL_CALL_4(22, 01, 32, 32, 32, 2)
KERNEL_CALL_4(22, 02, 32, 32, 32, 4)
KERNEL_CALL_4(22, 03, 32, 32, 32, 8)
KERNEL_CALL_4(22, 04, 32, 16, 16, 1)
KERNEL_CALL_4(22, 05, 64, 16, 16, 1)
KERNEL_CALL_4(22, 06, 64, 16, 16, 2)
KERNEL_CALL_4(22, 07, 64, 8, 8, 1)
KERNEL_CALL_4(22, 08, 64, 8, 8, 2)
KERNEL_CALL_4(22, 09, 128, 8, 8, 1)
KERNEL_CALL_4(22, 10, 128, 8, 8, 2)


// template <class T>
// void larft_gemv225(const SizeType m, const SizeType k, const T* v, const SizeType ldv, T* t,
//                    const SizeType ldt, whip::stream_t stream) {
//   constexpr unsigned kernel_tile_size_rows_v = 64;
//   constexpr unsigned kernel_tile_size_rows_t = 16;
//   constexpr unsigned kernel_tile_size_cols_t = 16;
//   constexpr unsigned cols = 1;
//
//   internal::larft_gemv22<kernel_tile_size_rows_v, kernel_tile_size_rows_t, kernel_tile_size_cols_t, cols>(
//       m, k, v, ldv, t, ldt, stream);
// }

DLAF_CUBLAS_LARFT_GEMV_ETI(, float);
DLAF_CUBLAS_LARFT_GEMV_ETI(, double);
DLAF_CUBLAS_LARFT_GEMV_ETI(, std::complex<float>);
DLAF_CUBLAS_LARFT_GEMV_ETI(, std::complex<double>);

}
